#include "GI2DRenderer.cuh"
#include "generic/Math.h"
#include "kernels/CudaVector.cuh"
#include "kernels/gi2d/CudaGI2DOverlay.cuh"
#include "kernels/gi2d/CudaGI2DPathTracer.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "kernels/gi2d/Curve.cuh"
#include "kernels/CudaRenderObjectContainer.cuh"
#include "kernels/CudaVector.cuh"
#include "kernels/gi2d/Tracable.cuh"
#include "kernels/gi2d/Curve.cuh"

using namespace Cuda;
using namespace GI2D;

GI2DRenderer::GI2DRenderer()
{
    m_uiGraph.DeclareState("kIdleState", this, &GI2DRenderer::OnIdleState);

    // Create tracable
    m_uiGraph.DeclareState("kCreateTracableOpen", this, &GI2DRenderer::OnCreateTracable);      
    m_uiGraph.DeclareState("kCreateTracableHover", this, &GI2DRenderer::OnCreateTracable);
    m_uiGraph.DeclareState("kCreateTracableAppend", this, &GI2DRenderer::OnCreateTracable);
    m_uiGraph.DeclareState("kCreateTracableClose", this, &GI2DRenderer::OnCreateTracable);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateTracableOpen", KeyboardButtonMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), nullptr, 0);
    m_uiGraph.DeclareAutoTransition("kCreateTracableOpen", "kCreateTracableHover");
    m_uiGraph.DeclareDeterministicTransition("kCreateTracableHover", "kCreateTracableHover", nullptr, nullptr, kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kCreateTracableHover", "kCreateTracableAppend", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0);
    //m_uiGraph.DeclareDeterministicTransition("kCreateTracableHover", "kCreateTracableAppend", nullptr, MouseButtonMap(kMouseLButton, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareAutoTransition("kCreateTracableAppend", "kCreateTracableHover");
    m_uiGraph.DeclareDeterministicTransition("kCreateTracableHover", "kCreateTracableClose", KeyboardButtonMap(VK_ESCAPE, kOnButtonDepressed), nullptr, 0);
    m_uiGraph.DeclareDeterministicTransition("kCreateTracableHover", "kCreateTracableClose", nullptr, MouseButtonMap(kMouseRButton, kOnButtonDepressed), 0);
    m_uiGraph.DeclareAutoTransition("kCreateTracableClose", "kIdleState");

    // Select/deselect tracable
    m_uiGraph.DeclareState("kSelectTracableDragging", this, &GI2DRenderer::OnSelectTracables);
    m_uiGraph.DeclareState("kSelectTracableEnd", this, &GI2DRenderer::OnSelectTracables);
    m_uiGraph.DeclareState("kDeselectTracable", this, &GI2DRenderer::OnSelectTracables);   
    m_uiGraph.DeclareNonDeterministicTransition("kIdleState", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0, this, &GI2DRenderer::DecideOnClickState);
    m_uiGraph.DeclareDeterministicTransition("kSelectTracableDragging", "kSelectTracableDragging", nullptr, MouseButtonMap(kMouseLButton, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kSelectTracableDragging", "kSelectTracableEnd", nullptr, MouseButtonMap(kMouseLButton, kOnButtonReleased), 0);
    m_uiGraph.DeclareAutoTransition("kSelectTracableEnd", "kIdleState");
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeselectTracable", nullptr, MouseButtonMap(kMouseRButton, kOnButtonDepressed), 0);
    m_uiGraph.DeclareAutoTransition("kDeselectTracable", "kIdleState");

    // Move tracable
    m_uiGraph.DeclareState("kMoveTracableBegin", this, &GI2DRenderer::OnMoveTracable);
    m_uiGraph.DeclareState("kMoveTracableDragging", this, &GI2DRenderer::OnMoveTracable);
    m_uiGraph.DeclareState("kMoveTracableEnd", this, &GI2DRenderer::OnMoveTracable);
    //m_uiGraph.DeclareNonDeterministicTransition("kIdleState", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0, this, &GI2DRenderer::DecideOnClickState);
    m_uiGraph.DeclareAutoTransition("kMoveTracableBegin", "kMoveTracableDragging");
    m_uiGraph.DeclareDeterministicTransition("kMoveTracableDragging", "kMoveTracableDragging", nullptr, MouseButtonMap(kMouseLButton, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kMoveTracableDragging", "kMoveTracableEnd", nullptr, MouseButtonMap(kMouseLButton, kOnButtonReleased), 0);
    m_uiGraph.DeclareAutoTransition("kMoveTracableEnd", "kIdleState");

    // Delete tracable
    m_uiGraph.DeclareState("kDeleteTracable", this, &GI2DRenderer::OnDeletePath);    
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeleteTracable", KeyboardButtonMap({ {VK_DELETE, kOnButtonDepressed}}), nullptr, 0);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeleteTracable", KeyboardButtonMap({ {VK_BACK, kOnButtonDepressed}}), nullptr, 0);
    m_uiGraph.DeclareAutoTransition("kDeleteTracable", "kIdleState");    

    m_uiGraph.Finalise();
}

GI2DRenderer::~GI2DRenderer()
{
    Destroy();
}

std::shared_ptr<RendererInterface> GI2DRenderer::Instantiate()
{
    return std::make_shared<GI2DRenderer>();
}

void GI2DRenderer::OnInitialise()
{
    m_viewCtx.transform = GI2D::ViewTransform2D(m_clientToNormMatrix, vec2(0.f), 0.f, 1.0f);
    m_viewCtx.dPdXY = length(vec2(m_viewCtx.transform.matrix.i00, m_viewCtx.transform.matrix.i10));
    m_viewCtx.zoomSpeed = 10.0f;
    m_viewCtx.sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));

    //m_primitiveContainer.Create(m_renderStream);
     
    m_renderObjects = CreateAsset<Cuda::RenderObjectContainer>(":gi2d/renderObjects");

    m_hostTracables = CreateAsset<Cuda::Host::AssetVector<GI2D::Host::Tracable>>(":gi2d/tracables", kVectorHostAlloc, m_renderStream);
    m_sceneBIH = CreateAsset<GI2D::Host::BIH2DAsset>(":gi2d/bih", 1);

    m_overlayRenderer = CreateAsset<GI2D::Host::Overlay>(":gi2d/overlay", m_sceneBIH, m_hostTracables, m_clientWidth, m_clientHeight, m_renderStream);
    m_pathTracer = CreateAsset<GI2D::Host::PathTracer>(":gi2d/pathTracer", m_sceneBIH, m_hostTracables, m_clientWidth, m_clientHeight, 2, m_renderStream);

    SetDirtyFlags(kGI2DDirtyAll);
}

void GI2DRenderer::Rebuild()
{
    std::lock_guard<std::mutex> lock(m_resourceMutex);
    
    if (m_dirtyFlags & kGI2DDirtyBVH)
    {
        // Rebuild and synchronise any tracables that were dirtied since the last iteration
        m_hostTracables->Clear();
        m_renderObjects->ForEachOfType<GI2D::Host::Tracable>([this](AssetHandle<GI2D::Host::Tracable>& tracable) -> bool
            {
                // Rebuild the tracable (it will decide whether any action needs to be taken)
                tracable->Rebuild(m_dirtyFlags, m_viewCtx);

                m_hostTracables->EmplaceBack(tracable);
                return true;
            });
        m_hostTracables->Synchronise(kVectorSyncUpload);

        // Cache the object bounding boxes
        /*m_tracableBBoxes.reserve(m_hostTracables->Size());
        for (auto& tracable : *m_hostTracables)
        {
            m_tracableBBoxes.emplace_back(tracable->GetBoundingBox());
        }*/

        // Create a tracable list ready for building
        // TODO: It's probably faster if we build on the already-sorted index list
        auto& primIdxs = m_sceneBIH->GetPrimitiveIndices();
        primIdxs.resize(m_hostTracables->Size());
        for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

        // Construct the BIH
        std::function<BBox2f(uint)> getPrimitiveBBox = [this](const uint& idx) -> BBox2f
        {
            return Grow((*m_hostTracables)[idx]->GetWorldSpaceBoundingBox(), 0.001f);
        };
        m_sceneBIH->Build(getPrimitiveBBox);
        //Log::Write("Rebuilt scene BIH: %s", m_sceneBIH->GetBoundingBox().Format());
    }

    // View has changed
    m_overlayRenderer->Rebuild(m_dirtyFlags, m_viewCtx, m_selectionCtx);
    m_pathTracer->Rebuild(m_dirtyFlags, m_viewCtx, m_selectionCtx);

    SetDirtyFlags(kGI2DDirtyAll, false);
}

void GI2DRenderer::OnDestroy()
{
    m_overlayRenderer.DestroyAsset();
    m_pathTracer.DestroyAsset();
    
    /*m_objects.overlayRenderer.DestroyAsset();
    m_objects.sceneBIH.DestroyAsset();
    m_objects.hostTracables.DestroyAsset();*/
}

uint GI2DRenderer::OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    //Log::Success("Back home!");
    return kUIStateOkay;
}

uint GI2DRenderer::OnDeletePath(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    if (m_selectionCtx.numSelected == 0) { return kUIStateOkay; }
    
    std::lock_guard <std::mutex> lock(m_resourceMutex);

    auto& tracables = *m_hostTracables;
    int emptyIdx = -1;
    int numDeleted = 0;
    for (int primIdx = 0; primIdx < tracables.Size(); ++primIdx)
    {
        if (tracables[primIdx]->IsSelected())
        {
            // Erase the object from the container
            m_renderObjects->Erase(tracables[primIdx]->GetAssetID());

            ++numDeleted;
            if (emptyIdx == -1) { emptyIdx = primIdx; }
        }
        else if (emptyIdx >= 0)
        {
            tracables[emptyIdx++] = tracables[primIdx];
        }
    }

    Assert(numDeleted <= tracables.Size());
    tracables.Resize(tracables.Size() - numDeleted);
    Log::Error("Delete!");

    m_selectionCtx.numSelected = 0;
    SetDirtyFlags(kGI2DDirtyBVH);    
    
    return kUIStateOkay;
}

uint GI2DRenderer::OnMoveTracable(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kMoveTracableBegin")
    {
        m_onMove.dragAnchor = m_viewCtx.mousePos;
    }
    else if (stateID == "kMoveTracableDragging")
    {
        // Update the selection overlay
        m_selectionCtx.selectedBBox += m_viewCtx.mousePos - m_onMove.dragAnchor;
        m_onMove.dragAnchor = m_viewCtx.mousePos;
        SetDirtyFlags(kGI2DDirtyUI);
    } 
    
    // Notify the scene objects of the move operation
    std::lock_guard <std::mutex> lock(m_resourceMutex);
    uint tracableDirtyFlags = 0u;
    for (auto& obj : *m_hostTracables)
    {
        if (obj->IsSelected())
        {
            // If the object has moved, trigger a rebuild of the BVH
            if (obj->OnMove(stateID, m_viewCtx) & kGI2DDirtyTransforms)
            {
                SetDirtyFlags(kGI2DDirtyBVH);
            }
        }
    }

    return kUIStateOkay;
}

void GI2DRenderer::DeselectAll()
{
    std::lock_guard <std::mutex> lock(m_resourceMutex);

    for (auto obj : *m_hostTracables) 
    { 
        obj->OnSelect(false); 
    }
    SetDirtyFlags(kGI2DDirtyUI);
    m_selectionCtx.numSelected = 0;
}

std::string GI2DRenderer::DecideOnClickState(const uint& sourceStateIdx)
{
    // If there are no paths selected, enter selection state. Otherwise, enter moving state.
    if (m_selectionCtx.numSelected == 0)
    {
        return "kSelectTracableDragging";
    }
    else
    {
        //Assert(selection.selectedBBox.HasValidArea());
        if (Grow(m_selectionCtx.selectedBBox, m_viewCtx.dPdXY * 2.f).Contains(m_viewCtx.mousePos))
        {
            return "kMoveTracableBegin";
        }
        else
        {
            // Deselect everything
            DeselectAll();
            return "kSelectTracableDragging";
        }
    }
    return "kSelectTracableDragging";
}

uint GI2DRenderer::OnSelectTracables(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kSelectTracableDragging")
    {
        auto& tracables = *m_hostTracables;
        const bool wasLassoing = m_selectionCtx.isLassoing;

        if (!m_selectionCtx.isLassoing)
        {
            // Deselect all the path segments
            DeselectAll();

            m_selectionCtx.mouseBBox = BBox2f(m_viewCtx.mousePos);
            m_selectionCtx.isLassoing = true;
        }

        m_selectionCtx.mouseBBox.upper = m_viewCtx.mousePos;
        m_selectionCtx.lassoBBox = Grow(Rectify(m_selectionCtx.mouseBBox), m_viewCtx.dPdXY * 2.);
        m_selectionCtx.selectedBBox = BBox2f::MakeInvalid();

        std::lock_guard <std::mutex> lock(m_resourceMutex);
        if (m_sceneBIH->IsConstructed())
        {
            const uint lastNumSelected = m_selectionCtx.numSelected;

            auto onIntersectPrim = [&tracables, this](const uint* primRange, const bool isInnerNode)
            {
                // Inner nodes are tested when the bounding box envelops them completely. Hence, there's no need to do a bbox checks.
                if (isInnerNode)
                {
                    for (int idx = primRange[0]; idx < primRange[1]; ++idx) { tracables[idx]->OnSelect(true); }
                    m_selectionCtx.numSelected += primRange[1] - primRange[0];
                }
                else
                {
                    for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                    {
                        const auto& bBoxWorld = tracables[idx]->GetWorldSpaceBoundingBox();
                        const bool isCaptured = bBoxWorld.Intersects(m_selectionCtx.lassoBBox);
                        if (isCaptured)
                        {
                            m_selectionCtx.selectedBBox = Union(m_selectionCtx.selectedBBox, bBoxWorld);
                            ++m_selectionCtx.numSelected;
                        }
                        tracables[idx]->OnSelect(isCaptured);
                    }
                }
            };
            m_sceneBIH->TestBBox(m_selectionCtx.lassoBBox, onIntersectPrim);

            // Only if the number of selected primitives has changed
            if (lastNumSelected != m_selectionCtx.numSelected)
            {
                if (m_selectionCtx.numSelected > 0 && !wasLassoing)
                {
                    m_selectionCtx.isLassoing = false;
                    m_uiGraph.SetState("kMoveTracableBegin");
                }
            }
        }

        SetDirtyFlags(kGI2DDirtyUI);
        //Log::Success("Selecting!");
    }
    else if (stateID == "kSelectTracableEnd")
    {
        m_selectionCtx.isLassoing = false;
        SetDirtyFlags(kGI2DDirtyUI);

        //Log::Success("Finished!");
    }
    else if (stateID == "kDeselectTracable")
    {
        for (auto& obj : *m_hostTracables) { obj->OnSelect(false); }
        SetDirtyFlags(kGI2DDirtyUI);

        //Log::Success("Finished!");
    }
    else
    {
        return kUIStateError;
    }

    return kUIStateOkay;
}

uint GI2DRenderer::OnCreateTracable(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    std::lock_guard <std::mutex> lock(m_resourceMutex);
    
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kCreateTracableOpen")
    {        
        //Create a new tracable and add it to the list of render objects
        m_onCreate.newObject = CreateAsset<GI2D::Host::Curve>(tfm::format("curve%i", m_renderObjects->GetUniqueIndex()));
        m_renderObjects->Emplace(AssetHandle<Cuda::Host::RenderObject>(m_onCreate.newObject), false);

        m_onCreate.newObject->OnCreate(stateID, m_viewCtx);
    }

    // Invoke the event handler of the new object
    SetDirtyFlags(m_onCreate.newObject->OnCreate(stateID, m_viewCtx));

    if (stateID == "kCreateTracableClose")
    {
        Assert(m_onCreate.newObject);

        // If the new object can't be finalised, delete it
        if (!m_onCreate.newObject->Finalise())
        {
            m_renderObjects->Erase(m_onCreate.newObject->GetAssetID());
            SetDirtyFlags(kGI2DDirtyBVH);

            Log::Success("Destroyed unfinalised tracable '%s'", m_onCreate.newObject->GetAssetID());
        }

        return kUIStateOkay;
    }    

    return kUIStateOkay;
}

void GI2DRenderer::OnRender()
{
    //std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //Log::Write("Tick");

    if (m_dirtyFlags)
    {
        Rebuild();
    }

    // Render the pass
    //m_objects.pathTracer->Render();
    m_overlayRenderer->Render();

    // If a blit is in progress, skip the composite step entirely.
    // TODO: Make this respond intelligently to frame rate. If the CUDA renderer is running at a lower FPS than the D3D renderer then it should wait rather than
    // than skipping frames like this.
    //m_renderSemaphore.Wait(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress);
    if (!m_renderSemaphore.Try(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress, false)) { return; }
    
    m_pathTracer->Composite(m_compositeImage);
    m_overlayRenderer->Composite(m_compositeImage);

    m_renderSemaphore.Try(kRenderManagerCompInProgress, kRenderManagerCompFinished, true);
}

void GI2DRenderer::OnKey(const uint code, const bool isSysKey, const bool isDown)
{

}

void GI2DRenderer::OnMouseButton(const uint code, const bool isDown)
{
    // Is the view being changed? 
    if (code == kMouseMButton)
    {
        m_viewCtx.dragAnchor = vec2(m_mouse.pos);
        m_viewCtx.rotAxis = normalize(m_viewCtx.dragAnchor - vec2(m_clientWidth, m_clientHeight) * 0.5f);
        m_viewCtx.transAnchor = m_viewCtx.transform.trans;
        m_viewCtx.scaleAnchor = m_viewCtx.transform.scale;
        m_viewCtx.rotAnchor = m_viewCtx.transform.rotate;
    }
}

void GI2DRenderer::OnMouseMove()
{
    // Dragging?
    if (IsMouseButtonDown(kMouseMButton))
    {
        OnViewChange();
    }

    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        m_viewCtx.mousePos = m_viewCtx.transform.matrix * vec2(m_mouse.pos);
    }
}

void GI2DRenderer::OnViewChange()
{
    auto& transform = m_viewCtx.transform;
    
    // Zooming?
    if (IsKeyDown(VK_CONTROL))
    {
        float logScaleAnchor = std::log2(::math::max(1e-10f, m_viewCtx.scaleAnchor));
        logScaleAnchor += m_viewCtx.zoomSpeed * float(m_mouse.pos.y - m_viewCtx.dragAnchor.y) / m_clientHeight;
        transform.scale = std::pow(2.0, logScaleAnchor);

        //Log::Write("Scale: %f", transform.scale);
    }
    // Rotating?
    else if (IsKeyDown(VK_SHIFT))
    {
        const vec2 delta = normalize(vec2(m_mouse.pos) - vec2(m_clientWidth, m_clientHeight) * 0.5f);
        const float theta = std::acos(dot(delta, m_viewCtx.rotAxis)) * (float(dot(delta, vec2(m_viewCtx.rotAxis.y, -m_viewCtx.rotAxis.x)) < 0.0f) * 2.0 - 1.0f);
        transform.rotate = m_viewCtx.rotAnchor + theta;

        if (std::abs(std::fmod(transform.rotate, kHalfPi)) < 0.05f) { transform.rotate = std::round(transform.rotate / kHalfPi) * kHalfPi; }

        //Log::Write("Theta: %f", transform.rotate);
    }
    // Translating
    else
    {
        // Update the transformation
        const mat3 newMat = ConstructViewMatrix(m_viewCtx.transAnchor, transform.rotate, transform.scale) * m_clientToNormMatrix;
        const vec2 dragDelta = (newMat * vec2(m_viewCtx.dragAnchor)) - (newMat * vec2(m_mouse.pos));
        transform.trans = m_viewCtx.transAnchor + dragDelta;

        //Log::Write("Trans: %s", m_viewCtx.trans.format());
    }

    // Update the parameters in the overlay renderer
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        transform.matrix = ConstructViewMatrix(transform.trans, transform.rotate, transform.scale) * m_clientToNormMatrix;
        m_viewCtx.Prepare();

        // Mark the scene as dirty
        SetDirtyFlags(kGI2DDirtyView);
    }
}

void GI2DRenderer::OnMouseWheel()
{

}

void GI2DRenderer::OnResizeClient()
{
}