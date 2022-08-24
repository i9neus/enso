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

GI2DRenderer::GI2DRenderer() : 
    m_viewCtx(m_resourceMutex)
{
    m_uiGraph.DeclareState("kIdleState", this, &GI2DRenderer::OnIdleState);

    // Create path
    m_uiGraph.DeclareState("kCreatePathOpen", this, &GI2DRenderer::OnCreateTracable);      
    m_uiGraph.DeclareState("kCreatePathHover", this, &GI2DRenderer::OnCreateTracable);
    m_uiGraph.DeclareState("kCreatePathAppend", this, &GI2DRenderer::OnCreateTracable);
    m_uiGraph.DeclareState("kCreatePathClose", this, &GI2DRenderer::OnCreateTracable);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreatePathOpen", KeyboardButtonMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), nullptr, 0);
    m_uiGraph.DeclareAutoTransition("kCreatePathOpen", "kCreatePathHover");
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathHover", nullptr, nullptr, kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathAppend", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0);
    //m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathAppend", nullptr, MouseButtonMap(kMouseLButton, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareAutoTransition("kCreatePathAppend", "kCreatePathHover");
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathClose", KeyboardButtonMap(VK_ESCAPE, kOnButtonDepressed), nullptr, 0);
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathClose", nullptr, MouseButtonMap(kMouseRButton, kOnButtonDepressed), 0);
    m_uiGraph.DeclareAutoTransition("kCreatePathClose", "kIdleState");

    // Select/deselect path
    m_uiGraph.DeclareState("kSelectPathDragging", this, &GI2DRenderer::OnSelectTracables);
    m_uiGraph.DeclareState("kSelectPathEnd", this, &GI2DRenderer::OnSelectTracables);
    m_uiGraph.DeclareState("kDeselectPath", this, &GI2DRenderer::OnSelectTracables);   
    m_uiGraph.DeclareNonDeterministicTransition("kIdleState", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0, this, &GI2DRenderer::DecideOnClickState);
    m_uiGraph.DeclareDeterministicTransition("kSelectPathDragging", "kSelectPathDragging", nullptr, MouseButtonMap(kMouseLButton, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kSelectPathDragging", "kSelectPathEnd", nullptr, MouseButtonMap(kMouseLButton, kOnButtonReleased), 0);
    m_uiGraph.DeclareAutoTransition("kSelectPathEnd", "kIdleState");
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeselectPath", nullptr, MouseButtonMap(kMouseRButton, kOnButtonDepressed), 0);
    m_uiGraph.DeclareAutoTransition("kDeselectPath", "kIdleState");

    // Move path
    m_uiGraph.DeclareState("kMovePathBegin", this, &GI2DRenderer::OnMoveTracable);
    m_uiGraph.DeclareState("kMovePathDragging", this, &GI2DRenderer::OnMoveTracable);
    //m_uiGraph.DeclareNonDeterministicTransition("kIdleState", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0, this, &GI2DRenderer::DecideOnClickState);
    m_uiGraph.DeclareAutoTransition("kMovePathBegin", "kMovePathDragging");
    m_uiGraph.DeclareDeterministicTransition("kMovePathDragging", "kMovePathDragging", nullptr, MouseButtonMap(kMouseLButton, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kMovePathDragging", "kIdleState", nullptr, MouseButtonMap(kMouseLButton, kOnButtonReleased), 0);

    // Delete path
    m_uiGraph.DeclareState("kDeletePath", this, &GI2DRenderer::OnDeletePath);    
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeletePath", KeyboardButtonMap({ {VK_DELETE, kOnButtonDepressed}}), nullptr, 0);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeletePath", KeyboardButtonMap({ {VK_BACK, kOnButtonDepressed}}), nullptr, 0);
    m_uiGraph.DeclareAutoTransition("kDeletePath", "kIdleState");    

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
    m_viewCtx.transform.dPdXY = length(vec2(m_viewCtx.transform.matrix.i00, m_viewCtx.transform.matrix.i10));
    m_viewCtx.zoomSpeed = 10.0f;
    m_viewCtx.transform.sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));

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
    
    if (m_dirtyFlags & kGI2DDirtyGeometry)
    {
        // Rebuild and synchronise any tracables that were dirtied since the last iteration
        m_hostTracables->Clear();
        m_renderObjects->ForEachOfType<GI2D::Host::Tracable>([this](AssetHandle<GI2D::Host::Tracable>& tracable) -> bool
            {
                if (tracable->GetDirtyFlags() & kGI2DDirtyGeometry)
                {
                    tracable->Rebuild();
                }
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
            return Grow((*m_hostTracables)[idx]->GetBoundingBox(), 0.001f);
        };
        m_sceneBIH->Build(getPrimitiveBBox);
        //Log::Write("Rebuilt scene BIH: %s", m_sceneBIH->GetBoundingBox().Format());
    }

    // View has changed
    m_overlayRenderer->Rebuild(m_dirtyFlags, m_viewCtx);
    m_pathTracer->Rebuild(m_dirtyFlags, m_viewCtx); 

    ClearDirtyFlags(kGI2DDirtyAll);
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
    /*if (m_objects.overlayParams.selection.numSelected != 0)
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        
        auto& segments = *m_objects.hostTracables;
        int emptyIdx = -1;
        int numDeleted = 0;
        for (int primIdx = 0; primIdx < segments.Size(); ++primIdx)
        {
            if (segments[primIdx].IsSelected())
            {
                ++numDeleted;
                if (emptyIdx == -1) { emptyIdx = primIdx; }
            }
            else if (emptyIdx >= 0)
            {
                segments[emptyIdx++] = segments[primIdx];
            }
        }

        Assert(numDeleted <= segments.Size());
        segments.Resize(segments.Size() - numDeleted);
        Log::Error("Delete!");

        m_objects.overlayParams.selection.numSelected = 0;
        SetDirtyFlags(kGI2DDirtyGeometry);
    }*/
    
    return kUIStateOkay;
}

uint GI2DRenderer::OnMoveTracable(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    /*const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kMovePathBegin")
    {
        // If the mouse isn't inside the object's bounding box, deselect it
        if (!Grow(m_objects.overlayParams.selection.selectedBBox, m_viewCtx.dPdXY * 2.f).Contains(m_viewCtx.mousePos))
        {
            m_uiGraph.SetState("kStateIdle");
        }
        else
        {
            m_moveTracable.dragAnchor = m_viewCtx.mousePos;
        }
    }
    else if (stateID == "kMovePathDragging")
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);

        const vec2 delta = m_viewCtx.mousePos - m_moveTracable.dragAnchor;
        for (auto& segment : *m_objects.hostTracables)
        {
            if (segment.IsSelected())
            {
                segment += delta;
            }
        }
        m_objects.overlayParams.selection.selectedBBox += delta;

        Log::Write("kMovePathDragging: %s, %s", m_moveTracable.dragAnchor.format(), m_viewCtx.mousePos.format());

        m_moveTracable.dragAnchor = m_viewCtx.mousePos;
        SetDirtyFlags(kGI2DDirtyAll);
    }
    else
    {
        return kUIStateError;
    }

    SetDirtyFlags(kGI2DDirtyParams);*/
    return kUIStateOkay;
}

std::string GI2DRenderer::DecideOnClickState(const uint& sourceStateIdx)
{
    // If there are no paths selected, enter selection state. Otherwise, enter moving state.
    /*auto& selection = m_objects.overlayParams.selection;
    if (selection.numSelected == 0)
    {
        return "kSelectPathDragging";
    }
    else
    {
        //Assert(selection.selectedBBox.HasValidArea());
        if (Grow(selection.selectedBBox, m_viewCtx.dPdXY * 2.f).Contains(m_viewCtx.mousePos))
        {
            return "kMovePathBegin";
        }
        else
        {
            // Deselect all the path segments
            std::lock_guard <std::mutex> lock(m_resourceMutex);
            for (auto segment : *m_objects.hostTracables) { segment.SetFlags(k2DPrimitiveSelected, false); }
            SetDirtyFlags(kGI2DDirtyPrimitiveAttributes);
            selection.numSelected = 0;

            return "kSelectPathDragging";
        }
    }*/
    return "kSelectPathDragging";
}

uint GI2DRenderer::OnSelectTracables(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    /*const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kSelectPathDragging")
    {
        auto& selection = m_objects.overlayParams.selection;
        auto& tracables = *m_objects.hostTracables;
        const bool wasLassoing = selection.isLassoing;

        if (!selection.isLassoing)
        {
            // Deselect all the path segments
            for (auto& tr : tracables) { tr->OnDeselect(); }

            selection.mouse0 = selection.mouse1 = m_viewCtx.mousePos;
            selection.isLassoing = true;
            selection.numSelected = 0;
            SetDirtyFlags(kGI2DDirtyPrimitiveAttributes);
        }

        selection.lassoBBox = Grow(Rectify(BBox2f(selection.mouse0, m_viewCtx.mousePos)), m_viewCtx.dPdXY * 2.);
        selection.selectedBBox = BBox2f::MakeInvalid();

        std::lock_guard <std::mutex> lock(m_resourceMutex);
        if (m_objects.sceneBIH->IsConstructed())
        {
            const uint lastNumSelected = selection.numSelected;

            auto onIntersectPrim = [&tracables, &selection](const uint& startIdx, const uint& endIdx, const bool isInnerNode)
            {
                // Inner nodes are tested when the bounding box envelops them completely. Hence, there's no need to do a bbox checks.
                if (isInnerNode)
                {
                    for (int idx = startIdx; idx < endIdx; ++idx) { tracables[idx].SetFlags(k2DPrimitiveSelected, true); }
                    selection.numSelected += endIdx - startIdx;
                }
                else
                {
                    for (int idx = startIdx; idx < endIdx; ++idx)
                    {
                        const bool isCaptured = tracables[idx].Intersects(selection.lassoBBox);
                        if (isCaptured)
                        {
                            selection.selectedBBox = Union(selection.selectedBBox, tracables[idx].GetBoundingBox());
                            ++selection.numSelected;
                        }
                        tracables[idx].SetFlags(k2DPrimitiveSelected, isCaptured);
                    }
                }
            };
            m_objects.sceneBIH->TestBBox(selection.lassoBBox, onIntersectPrim);

            // Only if the number of selected primitives has changed
            if (lastNumSelected != selection.numSelected)
            {
                if (selection.numSelected > 0 && !wasLassoing)
                {
                    selection.isLassoing = false;
                    m_uiGraph.SetState("kMovePathBegin");
                }

                SetDirtyFlags(kGI2DDirtyPrimitiveAttributes);
            }
        }

        Log::Success("Selecting!");
    }
    else if (stateID == "kSelectPathEnd")
    {
        m_objects.overlayParams.selection.isLassoing = false;
        SetDirtyFlags(kGI2DDirtyPrimitiveAttributes);

        Log::Success("Finished!");
    }
    else if (stateID == "kDeselectPath")
    {

        Log::Success("Finished!");
    }
    else
    {
        return kUIStateError;
    }

    SetDirtyFlags(kGI2DDirtyParams);*/
    return kUIStateOkay;
}

uint GI2DRenderer::OnCreateTracable(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    std::lock_guard <std::mutex> lock(m_resourceMutex);
    
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kCreatePathOpen")
    {        
        //Create a new tracable and add it to the list of render objects
        m_createObject.newObject = CreateAsset<GI2D::Host::Curve>(tfm::format("curve%i", m_renderObjects->GetUniqueIndex()));
        m_renderObjects->Emplace(AssetHandle<Cuda::Host::RenderObject>(m_createObject.newObject), false);

        m_createObject.newObject->OnCreate(stateID, m_viewCtx);   
    }

    // Invoke the event handler of the new object
    SetDirtyFlags(m_createObject.newObject->OnCreate(stateID, m_viewCtx));

    if (stateID == "kCreatePathClose")
    {
        Assert(m_createObject.newObject);
        
        // If the new object can't be finalised, delete it
        if (!m_createObject.newObject->Finalise())
        {
            m_renderObjects->Erase(m_createObject.newObject->GetAssetID());
            SetDirtyFlags(kGI2DDirtyGeometry);

            Log::Success("Destroyed unfinalised tracable '%s'", m_createObject.newObject->GetAssetID());
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
        transform.SetViewMatrix(ConstructViewMatrix(transform.trans, transform.rotate, transform.scale) * m_clientToNormMatrix);

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