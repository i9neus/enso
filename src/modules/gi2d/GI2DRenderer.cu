#include "GI2DRenderer.cuh"

#include "core/math/Math.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/GenericObjectContainer.cuh"
#include "core/Vector.cuh"
#include "core/Tuple.cuh"

#include "tracables/Tracable.cuh"
#include "tracables/Curve.cuh"
#include "lights/OmniLight.cuh"
#include "widgets/UIInspector.cuh" 
#include "SceneDescription.cuh"
#include "integrators/VoxelProxyGrid.cuh"
#include "layers/OverlayLayer.cuh"
#include "layers/PathTracerLayer.cuh"
#include "layers/VoxelProxyGridLayer.cuh"

#include "io/SerialisableObjectSchema.h"

//#include "kernels/gi2d/ObjectDebugger.cuh"

namespace Enso
{

    __host__ GI2DRenderer::GI2DRenderer(std::shared_ptr<CommandQueue> outQueue) :
        ModuleInterface(outQueue),
        m_isRunning(true)
    {
        // Load the object schema
        SerialisableObjectSchemaContainer::Load("schema.json");

        // Register the outbound commands
        m_outboundCmdQueue->RegisterCommand("OnCreateObject");
        m_outboundCmdQueue->RegisterCommand("OnUpdateObject");
        m_outboundCmdQueue->RegisterCommand("OnDeleteObject");

        // Register the inbound command handlers
        m_commandManager.RegisterEventHandler("OnUpdateObject", this, &GI2DRenderer::OnInboundUpdateObject);

        m_sceneObjectFactory.RegisterInstantiator<Host::Curve>(VirtualKeyMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());
        m_sceneObjectFactory.RegisterInstantiator<Host::OmniLight>(VirtualKeyMap({ {'W', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());

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

        // Utils
        m_uiGraph.DeclareState("kToggleRun", this, &GI2DRenderer::OnToggleRun);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kToggleRun", VirtualKeyMap({ {VK_SPACE, kOnButtonDepressed} }), 0);

        m_uiGraph.Finalise();
    }

    __host__ GI2DRenderer::~GI2DRenderer()
    {
        Destroy();
    }

    std::shared_ptr<ModuleInterface> GI2DRenderer::Instantiate(std::shared_ptr<CommandQueue> outQueue)
    {
        return std::make_shared<GI2DRenderer>(outQueue);
    }

    __host__ void GI2DRenderer::OnInitialise()
    {
        m_viewCtx.transform = ViewTransform2D(m_clientToNormMatrix, vec2(0.f), 0.f, 1.0f);
        m_viewCtx.dPdXY = length(vec2(m_viewCtx.transform.matrix.i00, m_viewCtx.transform.matrix.i10));
        m_viewCtx.zoomSpeed = 10.0f;
        m_viewCtx.sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));

        //m_primitiveContainer.Create(m_renderStream);

        m_sceneObjects = CreateAsset<GenericObjectContainer>(":gi2d/renderObjects");
        m_scene = CreateAsset<Host::SceneDescription>(":gi2d/sceneDescription");

        m_overlayRenderer = CreateAsset<Host::OverlayLayer>(":gi2d/overlay", m_scene, m_clientWidth, m_clientHeight, m_renderStream);
        m_pathTracerLayer = CreateAsset<Host::PathTracerLayer>(":gi2d/pathTracerLayer", m_scene, m_clientWidth, m_clientHeight, 2, m_renderStream);
        m_voxelProxyGridLayer = CreateAsset<Host::VoxelProxyGridLayer>(":gi2d/voxelProxyGridLayer", m_scene, 100, 100);

        //m_isosurfaceExplorer = CreateAsset<Host::IsosurfaceExplorer>(":gi2d/isosurfaceExplorer", m_scene, m_clientWidth, m_clientHeight, 1, m_renderStream);

        SetDirtyFlags(kDirtyAll);

        if (m_dirtyFlags)
        {
            Rebuild();
        }
    }

    __host__ void GI2DRenderer::OnDestroy()
    {
        m_overlayRenderer.DestroyAsset();
        m_pathTracerLayer.DestroyAsset();
        m_voxelProxyGridLayer.DestroyAsset();
        //m_isosurfaceExplorer.DestroyAsset();

        //m_scene->voxelProxy.DestroyAsset();

        m_scene.DestroyAsset();
        m_sceneObjects.DestroyAsset();
    }

    __host__ void GI2DRenderer::Rebuild()
    {
        std::lock_guard<std::mutex> lock(m_resourceMutex);

        if (!m_dirtyFlags) { return; }

        m_scene->Rebuild(m_sceneObjects, m_viewCtx, m_dirtyFlags);

        // View has changed
        m_overlayRenderer->Rebuild(m_dirtyFlags, m_viewCtx, m_selectionCtx);
        m_pathTracerLayer->Rebuild(m_dirtyFlags, m_viewCtx, m_selectionCtx);
        m_voxelProxyGridLayer->Rebuild(m_dirtyFlags, m_viewCtx, m_selectionCtx);
        //m_isosurfaceExplorer->Rebuild(m_dirtyFlags, m_viewCtx, m_selectionCtx);

        //m_scene->voxelProxy->Rebuild(m_dirtyFlags, m_viewCtx);

        SetDirtyFlags(kDirtyAll, false);
    }

    __host__ void GI2DRenderer::OnInboundUpdateObject(const Json::Node& node)
    {
        for (Json::Node::ConstIterator nodeIt = node.begin(); nodeIt != node.end(); ++nodeIt)
        {
            const std::string& objId = nodeIt.Name();
            auto objectHandle = m_sceneObjects->FindByID(objId);

            if (!objectHandle)
            {
                Log::Warning("Error: '%s' is not a valid scene object.", objId);
                continue;
            }

            const uint dirtyFlags = objectHandle->Deserialise(*nodeIt, Json::kRequiredWarn);
            SetDirtyFlags(dirtyFlags);
        }
    }

    __host__ void GI2DRenderer::EnqueueObjects(const std::string& eventId, const int flags, const AssetHandle<Host::SceneObject> asset)
    {
        if (!m_outboundCmdQueue->IsRegistered(eventId)) { return; }

        // Lambda to do the actual serialisation
        auto SerialiseImpl = [&](Json::Node& node, const AssetHandle<Host::SceneObject>& obj) -> void
        {
            // Create a new child object and add its class ID for the schema
            Json::Node childNode = node.AddChildObject(obj->GetAssetID());
            const std::string assetClass = obj->GetAssetClass();
            AssertMsgFmt(!assetClass.empty(), "Error: asset '%s' has no defined class", obj->GetAssetClass());
            childNode.AddValue("class", assetClass);

            // Deleted objects don't need their full attribute list serialised
            if (!(flags & kEnqueueIdOnly))
            {
                m_onCreate.newObject->Serialise(childNode, kSerialiseExposedOnly);
            }
        };

        Json::Node node = m_outboundCmdQueue->Create(eventId);
        if (flags & kEnqueueAll)
        {
            for (auto& obj : *m_sceneObjects) { SerialiseImpl(node, obj.DynamicCast<Host::SceneObject>()); }
        }
        else if (flags & kEnqueueSelected)
        {
            for (auto& obj : m_selectedTracables) { SerialiseImpl(node, obj.DynamicCast<Host::SceneObject>()); }
        }
        else if (flags & kEnqueueOne)
        {
            SerialiseImpl(node, asset);
        }

        m_outboundCmdQueue->Enqueue();  // Enqueue the staged command
    }

    __host__ uint GI2DRenderer::OnToggleRun(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        m_isRunning = !m_isRunning;
        Log::Warning(m_isRunning ? "Running" : "Paused");

        m_uiGraph.SetState("kIdleState");
        return kUIStateOkay;
    }

    __host__ uint GI2DRenderer::OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        //Log::Success("Back home!");
        return kUIStateOkay;
    }

    __host__ uint GI2DRenderer::OnDeleteSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        if (m_selectionCtx.numSelected == 0) { return kUIStateOkay; }

        std::lock_guard <std::mutex> lock(m_resourceMutex);

        auto& tracables = m_scene->Tracables();
        int emptyIdx = -1;
        int numDeleted = 0;
        for (int primIdx = 0; primIdx < tracables.Size(); ++primIdx)
        {
            if (tracables[primIdx]->IsSelected())
            {
                // Erase the object from the container
                m_sceneObjects->Erase(tracables[primIdx]->GetSceneObject().GetAssetID());

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

        EnqueueObjects("OnDeleteObject", kEnqueueSelected | kEnqueueIdOnly);

        // Clear the tracables list
        m_selectedTracables.clear();
        m_selectionCtx.numSelected = 0;

        SetDirtyFlags(kDirtyObjectBounds);

        return kUIStateOkay;
    }

    __host__ uint GI2DRenderer::OnMoveSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
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
            SetDirtyFlags(kDirtyUI);
        }

        // Notify the scene objects of the move operation  
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        uint tracableDirtyFlags = 0u;
        for (auto& obj : m_selectedTracables)
        {
            Assert(obj->IsSelected());

            // If the object has moved, trigger a rebuild of the BVH
            const uint objDirty = obj->OnMove(stateID, m_viewCtx);
            SetDirtyFlags(objDirty & (kDirtyObjectBounds | kDirtyObjectBVH));
        }

        // Enqueue the list of selected tracables
        EnqueueObjects("OnUpdateObject", kEnqueueSelected);

        return kUIStateOkay;
    }

    __host__ void GI2DRenderer::DeselectAll()
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);

        for (auto obj : m_scene->Tracables())
        {
            obj->OnSelect(false);
        }

        m_selectedTracables.clear();
        m_selectionCtx.numSelected = 0;

        SetDirtyFlags(kDirtyUI);
    }

    __host__ std::string GI2DRenderer::DecideOnClickState(const uint& sourceStateIdx)
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

    __host__ uint GI2DRenderer::OnSelectSceneObjects(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kSelectSceneObjectDragging")
        {
            auto& tracables = m_scene->Tracables();
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
            m_selectedTracables.clear();

            std::lock_guard <std::mutex> lock(m_resourceMutex);
            if (m_scene->TracableBIH().IsConstructed())
            {
                const uint lastNumSelected = m_selectionCtx.numSelected;

                auto onIntersectPrim = [&tracables, this](const uint* primRange, const bool isInnerNode)
                {
                    // Inner nodes are tested when the bounding box envelops them completely. Hence, there's no need to do a bbox checks.
                    if (isInnerNode)
                    {
                        for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                        {
                            m_selectedTracables.emplace_back(tracables[idx]);
                            tracables[idx]->OnSelect(true);
                        }
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
                                m_selectedTracables.emplace_back(tracables[idx]);
                                m_selectionCtx.selectedBBox = Union(m_selectionCtx.selectedBBox, bBoxWorld);
                                ++m_selectionCtx.numSelected;
                            }
                            tracables[idx]->OnSelect(isCaptured);
                        }
                    }
                };
                m_scene->TracableBIH().TestBBox(m_selectionCtx.lassoBBox, onIntersectPrim);

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

            SetDirtyFlags(kDirtyUI);
            //Log::Success("Selecting!");
        }
        else if (stateID == "kSelectSceneObjectEnd")
        {
            m_selectionCtx.isLassoing = false;
            SetDirtyFlags(kDirtyUI);

            //Log::Success("Finished!");
        }
        else if (stateID == "kDeselectSceneObject")
        {
            DeselectAll();
            SetDirtyFlags(kDirtyUI);

            //Log::Success("Finished!");
        }
        else
        {
            return kUIStateError;
        }

        return kUIStateOkay;
    }

    __host__ uint GI2DRenderer::OnCreateSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& trigger)
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);

        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kCreateSceneObjectOpen")
        {
            // Try and instante the objerct
            auto newObject = m_sceneObjectFactory.InstantiateFromHash(trigger.HashOf(), m_sceneObjects);
            m_onCreate.newObject = newObject.DynamicCast<Host::SceneObject>();
            Assert(m_onCreate.newObject);
        }

        // Invoke the event handler of the new object
        SetDirtyFlags(m_onCreate.newObject->OnCreate(stateID, m_viewCtx));

        // Some objects will automatically finalise themselves. If this happens, we're done.
        if (m_onCreate.newObject->IsFinalised())
        {
            EnqueueObjects("OnCreateObject", kEnqueueOne, m_onCreate.newObject);
            m_uiGraph.SetState("kIdleState");
            return kUIStateOkay;
        }

        if (stateID == "kCreateSceneObjectClose")
        {
            Assert(m_onCreate.newObject);

            // If the new object can't be finalised, delete it
            if (!m_onCreate.newObject->Finalise())
            {
                m_sceneObjects->Erase(m_onCreate.newObject->GetSceneObject().GetAssetID());
                SetDirtyFlags(kDirtyObjectBounds);

                Log::Success("Destroyed unfinalised tracable '%s'", m_onCreate.newObject->GetSceneObject().GetAssetID());
            }

            // Serialise the new object to the outbound queue
            EnqueueObjects("OnCreateObject", kEnqueueOne, m_onCreate.newObject);

            return kUIStateOkay;
        }

        return kUIStateOkay;
    }

    __host__ void GI2DRenderer::OnCommandsWaiting(CommandQueue& inbound)
    {
        m_commandManager.Flush(inbound, true);
    }

    __host__ void GI2DRenderer::OnRender()
    {
        if (m_dirtyFlags)
        {
            Rebuild();
        }

        //m_scene->voxelProxy->Render();

        // Render the pass
        //m_pathTracerLayer->Render();
        if (m_isRunning)
        {
            //if (m_renderTimer.Get() > 0.1f)
            {
                m_voxelProxyGridLayer->Render();
                //m_renderTimer.Reset();
                //Log::Write("-----");
            }
        }
        //m_isosurfaceExplorer->Render();
        m_overlayRenderer->Render();

        // If a blit is in progress, skip the composite step entirely.
        // TODO: Make this respond intelligently to frame rate. If the CUDA renderer is running at a lower FPS than the D3D renderer then it should wait rather than
        // than skipping frames like this.
        //m_renderSemaphore.Wait(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress);
        if (!m_renderSemaphore.Try(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress, false)) { return; }

        //m_compositeImage->Clear(vec4(kZero, 1.0f));
        //m_pathTracerLayer->Composite(m_compositeImage);
        m_voxelProxyGridLayer->Composite(m_compositeImage);
        //m_isosurfaceExplorer->Composite(m_compositeImage);    
        m_overlayRenderer->Composite(m_compositeImage);

        m_renderSemaphore.Try(kRenderManagerCompInProgress, kRenderManagerCompFinished, true);
    }

    __host__ void GI2DRenderer::OnKey(const uint code, const bool isSysKey, const bool isDown)
    {

    }

    __host__ void GI2DRenderer::OnMouseButton(const uint code, const bool isDown)
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

    __host__ void GI2DRenderer::OnMouseMove()
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

    __host__ void GI2DRenderer::OnViewChange()
    {
        auto& transform = m_viewCtx.transform;

        // Zooming?
        if (IsKeyDown(VK_CONTROL))
        {
            float logScaleAnchor = std::log2(std::max(1e-10f, m_viewCtx.scaleAnchor));
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
            SetDirtyFlags(kDirtyView);
        }
    }

    __host__ void GI2DRenderer::OnMouseWheel()
    {

    }

    __host__ void GI2DRenderer::OnResizeClient()
    {
    } 

    __host__ bool GI2DRenderer::Serialise(Json::Document& json, const int flags)
    {
        return true;
    }
}