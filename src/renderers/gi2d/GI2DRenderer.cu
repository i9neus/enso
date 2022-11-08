#include "GI2DRenderer.cuh"
#include "kernels/CudaVector.cuh"
#include "kernels/gi2d/layers/OverlayLayer.cuh"
#include "kernels/gi2d/layers/PathTracerLayer.cuh"
//#include "kernels/gi2d/layers/GI2DIsosurfaceExplorer.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "kernels/CudaRenderObjectContainer.cuh"
#include "kernels/CudaVector.cuh"
#include "kernels/gi2d/tracables/Tracable.cuh"
#include "kernels/gi2d/tracables/Curve.cuh"
//#include "kernels/gi2d/widgets/UIInspector.cuh"
#include "kernels/Tuple.cuh"

#include "kernels/gi2d/integrators/VoxelProxyGrid.cuh"

//#include "kernels/gi2d/ObjectDebugger.cuh"

using namespace Cuda;
using namespace GI2D;

GI2DRenderer::GI2DRenderer()
{
    // Declare the scene object instantiators
    // TODO: Merge this code with RenderObjectFactory
    AddInstantiator<GI2D::Host::Curve>('Q');
    //AddInstantiator<GI2D::Host::UIInspector>('W');

    m_uiGraph.DeclareState("kIdleState", this, &GI2DRenderer::OnIdleState);

    // Create scene object
    m_uiGraph.DeclareState("kCreateSceneObjectOpen", this, &GI2DRenderer::OnCreateSceneObject);      
    m_uiGraph.DeclareState("kCreateSceneObjectHover", this, &GI2DRenderer::OnCreateSceneObject);
    m_uiGraph.DeclareState("kCreateSceneObjectAppend", this, &GI2DRenderer::OnCreateSceneObject);
    m_uiGraph.DeclareState("kCreateSceneObjectClose", this, &GI2DRenderer::OnCreateSceneObject);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateSceneObjectOpen", VirtualKeyMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateSceneObjectOpen", VirtualKeyMap({ {'W', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
    m_uiGraph.DeclareDeterministicAutoTransition("kCreateSceneObjectOpen", "kCreateSceneObjectHover");
    m_uiGraph.DeclareDeterministicTransition("kCreateSceneObjectHover", "kCreateSceneObjectHover", nullptr, kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kCreateSceneObjectHover", "kCreateSceneObjectAppend", VirtualKeyMap(VK_LBUTTON, kOnButtonDepressed), 0);
    m_uiGraph.DeclareDeterministicAutoTransition("kCreateSceneObjectAppend", "kCreateSceneObjectHover");
    m_uiGraph.DeclareDeterministicTransition("kCreateSceneObjectHover", "kCreateSceneObjectClose", VirtualKeyMap(VK_RBUTTON, kOnButtonDepressed), 0);
    m_uiGraph.DeclareDeterministicAutoTransition("kCreateSceneObjectClose", "kIdleState");

    // Select/deselect scene object
    m_uiGraph.DeclareState("kSelectSceneObjectDragging", this, &GI2DRenderer::OnSelectSceneObjects);
    m_uiGraph.DeclareState("kSelectSceneObjectEnd", this, &GI2DRenderer::OnSelectSceneObjects);
    m_uiGraph.DeclareState("kDeselectSceneObject", this, &GI2DRenderer::OnSelectSceneObjects);
    m_uiGraph.DeclareNonDeterministicTransition("kIdleState", VirtualKeyMap(VK_LBUTTON, kOnButtonDepressed), 0, this, &GI2DRenderer::DecideOnClickState);
    m_uiGraph.DeclareDeterministicTransition("kSelectSceneObjectDragging", "kSelectSceneObjectDragging", VirtualKeyMap(VK_LBUTTON, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kSelectSceneObjectDragging", "kSelectSceneObjectEnd", VirtualKeyMap(VK_LBUTTON, kOnButtonReleased), 0);
    m_uiGraph.DeclareDeterministicAutoTransition("kSelectSceneObjectEnd", "kIdleState");
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeselectSceneObject", VirtualKeyMap(VK_RBUTTON, kOnButtonDepressed), 0);
    m_uiGraph.DeclareDeterministicAutoTransition("kDeselectSceneObject", "kIdleState");

    // Move scene object
    m_uiGraph.DeclareState("kMoveSceneObjectBegin", this, &GI2DRenderer::OnMoveSceneObject);
    m_uiGraph.DeclareState("kMoveSceneObjectDragging", this, &GI2DRenderer::OnMoveSceneObject);
    m_uiGraph.DeclareState("kMoveSceneObjectEnd", this, &GI2DRenderer::OnMoveSceneObject);
    //m_uiGraph.DeclareNonDeterministicTransition("kIdleState", nullptr, MouseButtonMap(VK_LBUTTON, kOnButtonDepressed), 0, this, &GI2DRenderer::DecideOnClickState);
    m_uiGraph.DeclareDeterministicAutoTransition("kMoveSceneObjectBegin", "kMoveSceneObjectDragging");
    m_uiGraph.DeclareDeterministicTransition("kMoveSceneObjectDragging", "kMoveSceneObjectDragging", VirtualKeyMap(VK_LBUTTON, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kMoveSceneObjectDragging", "kMoveSceneObjectEnd", VirtualKeyMap(VK_LBUTTON, kOnButtonReleased), 0);
    m_uiGraph.DeclareDeterministicAutoTransition("kMoveSceneObjectEnd", "kIdleState");

    // Delete scene object
    m_uiGraph.DeclareState("kDeleteSceneObjects", this, &GI2DRenderer::OnDeleteSceneObject);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeleteSceneObjects", VirtualKeyMap({ {VK_DELETE, kOnButtonDepressed} }), 0);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeleteSceneObjects", VirtualKeyMap({ {VK_BACK, kOnButtonDepressed} }), 0);
    m_uiGraph.DeclareDeterministicAutoTransition("kDeleteSceneObjects", "kIdleState");

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
    m_scene = CreateAsset<GI2D::Host::SceneDescription>(":gi2d/sceneDescription");

    m_scene->hostTracables = CreateAsset<GI2D::Host::TracableContainer>(":gi2d/tracables", Core::kVectorHostAlloc);
    m_scene->hostInspectors = CreateAsset<GI2D::Host::InspectorContainer>(":gi2d/inspectors", Core::kVectorHostAlloc);
    m_scene->sceneBIH = CreateAsset<GI2D::Host::BIH2DAsset>(":gi2d/bih", 1);
    m_scene->voxelProxy = CreateAsset<GI2D::Host::VoxelProxyGrid>(":gi2d/voxelproxy", m_scene, 100, 100);
    m_scene->Prepare();

    m_overlayRenderer = CreateAsset<GI2D::Host::OverlayLayer>(":gi2d/overlay", m_scene, m_clientWidth, m_clientHeight, m_renderStream);
    m_pathTracerLayer = CreateAsset<GI2D::Host::PathTracerLayer>(":gi2d/pathTracerLayer", m_scene, m_clientWidth, m_clientHeight, 2, m_renderStream);
    //m_isosurfaceExplorer = CreateAsset<GI2D::Host::IsosurfaceExplorer>(":gi2d/isosurfaceExplorer", m_scene, m_clientWidth, m_clientHeight, 1, m_renderStream);

    SetDirtyFlags(kGI2DDirtyAll);

    if (m_dirtyFlags)
    {
        Rebuild();
    }
}

void GI2DRenderer::OnDestroy()
{
    m_overlayRenderer.DestroyAsset();
    m_pathTracerLayer.DestroyAsset();
    //m_isosurfaceExplorer.DestroyAsset();

    m_scene->voxelProxy.DestroyAsset();

    m_scene->hostTracables.DestroyAsset();
    m_scene->hostInspectors.DestroyAsset();
    m_scene->sceneBIH.DestroyAsset();

    m_renderObjects.DestroyAsset();
    m_scene.DestroyAsset();
}

void GI2DRenderer::Rebuild()
{
    std::lock_guard<std::mutex> lock(m_resourceMutex);

    if (m_dirtyFlags & kGI2DDirtyBVH)
    {
        // Rebuild and synchronise any tracables that were dirtied since the last iteration
        m_scene->hostTracables->Clear();
        m_renderObjects->ForEachOfType<GI2D::Host::TracableInterface>([this](AssetHandle<GI2D::Host::TracableInterface>& tracable) -> bool
            {
                // Rebuild the tracable (it will decide whether any action needs to be taken)
                if (tracable->Rebuild(m_dirtyFlags, m_viewCtx))
                {
                    m_scene->hostTracables->EmplaceBack(tracable);
                }

                return true;
            });        
        m_scene->hostTracables->Synchronise(Core::kVectorSyncUpload);

        /*m_scene->hostInspectors->Clear();
        m_renderObjects->ForEachOfType<GI2D::Host::UIInspector>([this](AssetHandle<GI2D::Host::UIInspector>& widget) -> bool
            {
                m_scene->hostInspectors->EmplaceBack(widget);
                return true;
            });
        m_scene->hostInspectors->Synchronise(Core::kVectorSyncUpload);*/

        // Cache the object bounding boxes
        /*m_tracableBBoxes.reserve(m_scene->hostTracables->Size());
        for (auto& tracable : *m_scene->hostTracables)
        {
            m_tracableBBoxes.emplace_back(tracable->GetBoundingBox());
        }*/

        // Create a tracable list ready for building
        // TODO: It's probably faster if we build on the already-sorted index list
        auto& primIdxs = m_scene->sceneBIH->GetPrimitiveIndices();
        primIdxs.resize(m_scene->hostTracables->Size());
        for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

        // Construct the BIH
        std::function<BBox2f(uint)> getPrimitiveBBox = [this](const uint& idx) -> BBox2f
        {
            return Grow((*m_scene->hostTracables)[idx]->GetWorldSpaceBoundingBox(), 0.001f);
        };
        m_scene->sceneBIH->Build(getPrimitiveBBox);
        //Log::Write("Rebuilt scene BIH: %s", m_scene->sceneBIH->GetBoundingBox().Format());
    }

    /*if (m_dirtyFlags & kGI2DDirtyTransforms)
    {
        m_scene->hostInspectors->Clear();
        m_renderObjects->ForEachOfType<GI2D::Host::UIInspector>([this](AssetHandle<GI2D::Host::UIInspector>& widget) -> bool
            {
                if (widget->Rebuild(m_dirtyFlags, m_viewCtx))
                {
                    m_scene->hostInspectors->EmplaceBack(widget);
                }

                return true;
            });
        m_scene->hostInspectors->Synchronise(kVectorSyncUpload);
    }*/

    // View has changed
    m_overlayRenderer->Rebuild(m_dirtyFlags, m_viewCtx, m_selectionCtx);
    m_pathTracerLayer->Rebuild(m_dirtyFlags, m_viewCtx, m_selectionCtx);
    //m_isosurfaceExplorer->Rebuild(m_dirtyFlags, m_viewCtx, m_selectionCtx);
    m_scene->voxelProxy->Rebuild(m_dirtyFlags, m_viewCtx);

    SetDirtyFlags(kGI2DDirtyAll, false);
}

uint GI2DRenderer::OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
{
    //Log::Success("Back home!");
    return kUIStateOkay;
}

uint GI2DRenderer::OnDeleteSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
{
    if (m_selectionCtx.numSelected == 0) { return kUIStateOkay; }

    std::lock_guard <std::mutex> lock(m_resourceMutex);

    auto& tracables = *m_scene->hostTracables;
    int emptyIdx = -1;
    int numDeleted = 0;
    for (int primIdx = 0; primIdx < tracables.Size(); ++primIdx)
    {
        if (tracables[primIdx]->IsSelected())
        {
            // Erase the object from the container
            m_renderObjects->Erase(tracables[primIdx]->GetRenderObject().GetAssetID());

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

uint GI2DRenderer::OnMoveSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
{
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kMoveSceneObjectBegin")
    {
        m_onMove.dragAnchor = m_viewCtx.mousePos;
    }
    else if (stateID == "kMoveSceneObjectDragging")
    {
        // Update the selection overlay
        m_selectionCtx.selectedBBox += m_viewCtx.mousePos - m_onMove.dragAnchor;
        m_onMove.dragAnchor = m_viewCtx.mousePos;
        SetDirtyFlags(kGI2DDirtyUI);
    }

    // Notify the scene objects of the move operation
    std::lock_guard <std::mutex> lock(m_resourceMutex);
    uint tracableDirtyFlags = 0u;
    for (auto& obj : *m_scene->hostTracables)
    {
        if (obj->IsSelected())
        {
            // If the object has moved, trigger a rebuild of the BVH
            if (obj->OnMove(stateID, m_viewCtx) & (kGI2DDirtyTransforms | kGI2DDirtyBVH))
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

    for (auto obj : *m_scene->hostTracables)
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
        return "kSelectSceneObjectDragging";
    }
    else
    {
        //Assert(selection.selectedBBox.HasValidArea());
        if (Grow(m_selectionCtx.selectedBBox, m_viewCtx.dPdXY * 2.f).Contains(m_viewCtx.mousePos))
        {
            return "kMoveSceneObjectBegin";
        }
        else
        {
            // Deselect everything
            DeselectAll();
            return "kSelectSceneObjectDragging";
        }
    }
    return "kSelectSceneObjectDragging";
}

uint GI2DRenderer::OnSelectSceneObjects(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
{
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kSelectSceneObjectDragging")
    {
        auto& tracables = *m_scene->hostTracables;
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
        if (m_scene->sceneBIH->IsConstructed())
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
            m_scene->sceneBIH->TestBBox(m_selectionCtx.lassoBBox, onIntersectPrim);

            // Only if the number of selected primitives has changed
            if (lastNumSelected != m_selectionCtx.numSelected)
            {
                if (m_selectionCtx.numSelected > 0 && !wasLassoing)
                {
                    m_selectionCtx.isLassoing = false;
                    m_uiGraph.SetState("kMoveSceneObjectBegin");
                }
            }
        }

        SetDirtyFlags(kGI2DDirtyUI);
        //Log::Success("Selecting!");
    }
    else if (stateID == "kSelectSceneObjectEnd")
    {
        m_selectionCtx.isLassoing = false;
        SetDirtyFlags(kGI2DDirtyUI);

        //Log::Success("Finished!");
    }
    else if (stateID == "kDeselectSceneObject")
    {
        DeselectAll();
        SetDirtyFlags(kGI2DDirtyUI);

        //Log::Success("Finished!");
    }
    else
    {
        return kUIStateError;
    }

    return kUIStateOkay;
}

uint GI2DRenderer::OnCreateSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& trigger)
{
    std::lock_guard <std::mutex> lock(m_resourceMutex);
    
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kCreateSceneObjectOpen")
    {        
        //Create a new tracable and add it to the list of render objects
        //m_onCreate.newObject = CreateAsset<GI2D::Host::Curve>(tfm::format("curve%i", m_renderObjects->GetUniqueIndex()));
       
        if (trigger.IsSet('Q'))
        {
            auto newObject = CreateAsset<GI2D::Host::Curve>(tfm::format("curve%i", m_renderObjects->GetUniqueIndex()));
            m_renderObjects->Emplace(AssetHandle<Cuda::Host::RenderObject>(newObject), false);
            m_onCreate.newObject = newObject;
        }
        /*else if (trigger.IsSet('W'))
        {
            auto newObject = CreateAsset<GI2D::Host::UIInspector>(tfm::format("inspector%i", m_renderObjects->GetUniqueIndex()));
            m_renderObjects->Emplace(AssetHandle<Cuda::Host::RenderObject>(newObject), false);
            m_onCreate.newObject = newObject;
        }*/
        else { AssertMsg(false, "Invalid trigger"); }               

        m_onCreate.newObject->OnCreate(stateID, m_viewCtx);
    }

    // Invoke the event handler of the new object
    SetDirtyFlags(m_onCreate.newObject->OnCreate(stateID, m_viewCtx));

    // Some objects will automatically finalise themselves. If this happens, we're done.
    if (m_onCreate.newObject->IsFinalised())
    {
        m_uiGraph.SetState("kIdleState");
        return kUIStateOkay;
    }

    if (stateID == "kCreateSceneObjectClose")
    {
        Assert(m_onCreate.newObject);

        // If the new object can't be finalised, delete it
        if (!m_onCreate.newObject->Finalise())
        {
            m_renderObjects->Erase(m_onCreate.newObject->GetRenderObject().GetAssetID());
            SetDirtyFlags(kGI2DDirtyBVH);

            Log::Success("Destroyed unfinalised tracable '%s'", m_onCreate.newObject->GetRenderObject().GetAssetID());
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

    m_scene->voxelProxy->Render();

    // Render the pass
    m_pathTracerLayer->Render();
    //m_isosurfaceExplorer->Render();
    m_overlayRenderer->Render();

    // If a blit is in progress, skip the composite step entirely.
    // TODO: Make this respond intelligently to frame rate. If the CUDA renderer is running at a lower FPS than the D3D renderer then it should wait rather than
    // than skipping frames like this.
    //m_renderSemaphore.Wait(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress);
    if (!m_renderSemaphore.Try(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress, false)) { return; }
    
    //m_compositeImage->Clear(vec4(kZero, 1.0f));
    m_pathTracerLayer->Composite(m_compositeImage);
    //m_isosurfaceExplorer->Composite(m_compositeImage);    
    m_overlayRenderer->Composite(m_compositeImage);

    m_renderSemaphore.Try(kRenderManagerCompInProgress, kRenderManagerCompFinished, true);
}

void GI2DRenderer::OnKey(const uint code, const bool isSysKey, const bool isDown)
{

}

void GI2DRenderer::OnMouseButton(const uint code, const bool isDown)
{
    // Is the view being changed? 
    if (code == VK_MBUTTON)
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
    if (IsMouseButtonDown(VK_MBUTTON))
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
        float logScaleAnchor = std::log2(::max(1e-10f, m_viewCtx.scaleAnchor));
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