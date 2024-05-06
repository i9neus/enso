#include "GI2DRenderer.cuh"

#include "core/math/Math.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/GenericObjectContainer.cuh"
#include "core/Vector.cuh"
#include "core/Tuple.cuh"

#include "tracables/LineStrip.cuh"
#include "tracables/KIFS.cuh"

#include "lights/OmniLight.cuh"

#include "integrators/PerspectiveCamera.cuh"
#include "integrators/VoxelProxyGrid.cuh"

#include "scene/SceneBuilder.cuh"
#include "layers/OverlayLayer.cuh"

#include "io/SerialisableObjectSchema.h"

//#include "kernels/gi2d/ObjectDebugger.cuh"

namespace Enso
{

    __host__ Host::GI2DRenderer::GI2DRenderer(const InitCtx& initCtx, std::shared_ptr<CommandQueue> outQueue) :
        Dirtyable(initCtx),
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
        m_commandManager.RegisterEventHandler("OnUpdateObject", this, &Host::GI2DRenderer::OnInboundUpdateObject);

        // Register the scene object instantiators
        RegisterInstantiators();

        // Declare the transition graph that will drive the UI
        DeclareStateTransitionGraph();   

        // Declare any listeners that will respond to changes in the scene graph
        DeclareListeners();
    }

    __host__ Host::GI2DRenderer::~GI2DRenderer() noexcept
    {
        m_overlayRenderer.DestroyAsset();
        m_voxelProxyGrid.DestroyAsset();
        
        m_sceneBuilder.DestroyAsset();
        m_sceneContainer.DestroyAsset();
    }

    __host__ void Host::GI2DRenderer::RegisterInstantiators()
    {
        m_sceneObjectFactory.RegisterInstantiator<Host::LineStrip>(VirtualKeyMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());
        m_sceneObjectFactory.RegisterInstantiator<Host::KIFS>(VirtualKeyMap({ {'W', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());
        m_sceneObjectFactory.RegisterInstantiator<Host::PerspectiveCamera>(VirtualKeyMap({ {'E', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());
        m_sceneObjectFactory.RegisterInstantiator<Host::OmniLight>(VirtualKeyMap({ {'A', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());

        //m_sceneObjectFactory.RegisterInstantiator<Host::OverlayLayer>();
        //m_sceneObjectFactory.RegisterInstantiator<Host::VoxelProxyGrid>();
    }

    __host__ void Host::GI2DRenderer::DeclareStateTransitionGraph()
    {
        m_uiGraph.DeclareState("kIdleState", this, &Host::GI2DRenderer::OnIdleState);

        // Create scene object
        m_uiGraph.DeclareState("kCreateSceneObjectOpen", this, &GI2DRenderer::OnCreateSceneObject);
        m_uiGraph.DeclareState("kCreateSceneObjectHover", this, &GI2DRenderer::OnCreateSceneObject);
        m_uiGraph.DeclareState("kCreateSceneObjectAppend", this, &GI2DRenderer::OnCreateSceneObject);
        m_uiGraph.DeclareState("kCreateSceneObjectClose", this, &GI2DRenderer::OnCreateSceneObject);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateSceneObjectOpen", VirtualKeyMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateSceneObjectOpen", VirtualKeyMap({ {'A', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateSceneObjectOpen", VirtualKeyMap({ {'E', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateSceneObjectOpen", VirtualKeyMap({ {'W', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kCreateSceneObjectOpen", "kCreateSceneObjectHover");
        m_uiGraph.DeclareDeterministicTransition("kCreateSceneObjectHover", "kCreateSceneObjectHover", VirtualKeyMap::Nothing(), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kCreateSceneObjectHover", "kCreateSceneObjectAppend", VirtualKeyMap(VK_LBUTTON, kOnButtonDepressed), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kCreateSceneObjectAppend", "kCreateSceneObjectHover");
        m_uiGraph.DeclareDeterministicTransition("kCreateSceneObjectHover", "kCreateSceneObjectClose", VirtualKeyMap(VK_RBUTTON, kOnButtonDepressed), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kCreateSceneObjectClose", "kIdleState");

        // Delegate mouse actions to scene objects
        m_uiGraph.DeclareState("kDelegateSceneObjectBegin", this, &GI2DRenderer::OnDelegateSceneObject);
        m_uiGraph.DeclareState("kDelegateSceneObjectEnd", this, &GI2DRenderer::OnDelegateSceneObject);
        m_uiGraph.DeclareState("kDelegateSceneObjectDragging", this, &GI2DRenderer::OnDelegateSceneObject);
        m_uiGraph.DeclareDeterministicAutoTransition("kDelegateSceneObjectBegin", "kDelegateSceneObjectDragging");
        m_uiGraph.DeclareDeterministicTransition("kDelegateSceneObjectDragging", "kDelegateSceneObjectDragging", VirtualKeyMap(VK_LBUTTON, kButtonDown), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kDelegateSceneObjectDragging", "kDelegateSceneObjectEnd", VirtualKeyMap(VK_LBUTTON, kOnButtonReleased), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kDelegateSceneObjectEnd", "kIdleState");

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

    __host__ void Host::GI2DRenderer::DeclareListeners()
    {
        Listen({ kDirtyObjectBoundingBox });
    }

    __host__ void Host::GI2DRenderer::OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller)
    {

    }

    std::shared_ptr<ModuleInterface> Host::GI2DRenderer::Instantiate(std::shared_ptr<CommandQueue> outQueue)
    {
        AssetHandle<ModuleInterface> newAsset = AssetAllocator::CreateAsset<Host::GI2DRenderer>("gi2d", outQueue);
        return std::shared_ptr<ModuleInterface>(newAsset);
    }

    __host__ void Host::GI2DRenderer::OnInitialise()
    {
        m_viewCtx.transform = ViewTransform2D(m_clientToNormMatrix, vec2(0.f), 0.f, 1.0f);
        m_viewCtx.dPdXY = length(vec2(m_viewCtx.transform.matrix.i00, m_viewCtx.transform.matrix.i10));
        m_viewCtx.zoomSpeed = 10.0f;
        m_viewCtx.sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));

        LoadScene();

        m_sceneBuilder->Rebuild(true);
    }

    __host__ void Host::GI2DRenderer::LoadScene()
    {
        m_sceneContainer = AssetAllocator::CreateChildAsset<Host::SceneContainer>(*this, "sceneContainer");
        m_sceneBuilder = AssetAllocator::CreateChildAsset<Host::SceneBuilder>(*this, "sceneBuilder", m_sceneContainer);

        // Create some default scene objects
        Json::Node emptyDocument;
        m_overlayRenderer = AssetAllocator::CreateChildAsset<Host::OverlayLayer>(*this, "overlayLayer", m_sceneContainer, m_clientWidth, m_clientHeight, m_renderStream);
        m_voxelProxyGrid = AssetAllocator::CreateChildAsset<Host::VoxelProxyGrid>(*this, "voxelProxyGridLayer", emptyDocument, m_sceneContainer);

        // Emplace them into the scene object list and enqueue them
        m_sceneContainer->Emplace(m_overlayRenderer.StaticCast<Host::GenericObject>());
        m_sceneContainer->Emplace(m_voxelProxyGrid.StaticCast<Host::GenericObject>());
        EnqueueOutboundSerialisation("OnCreateObject", kEnqueueAll);
    }

    __host__ void Host::GI2DRenderer::OnInboundUpdateObject(const Json::Node& node)
    {
        for (Json::Node::ConstIterator nodeIt = node.begin(); nodeIt != node.end(); ++nodeIt)
        {
            const std::string& objId = nodeIt.Name();
            auto objectHandle = m_sceneContainer->GenericObjects().FindByID(objId);

            if (!objectHandle)
            {
                Log::Warning("Error: '%s' is not a valid scene object.", objId);
                continue;
            }

            objectHandle->Deserialise(*nodeIt, Json::kRequiredWarn);
        }
    }

    __host__ void Host::GI2DRenderer::EnqueueOutboundSerialisation(const std::string& eventId, const int flags, const AssetHandle<Host::GenericObject> asset)
    {
        // TODO: This should be smarter and happen automatically whenever an object is dirtied. 
        
        if (!m_outboundCmdQueue->IsRegistered(eventId)) { return; }

        // Lambda to do the actual serialisation
        auto SerialiseImpl = [&](Json::Node& node, const AssetHandle<Host::GenericObject>& obj) -> void
        {           
            // Create a new child object and add its class ID for the schema
            Json::Node childNode = node.AddChildObject(obj->GetAssetID(), Json::kPathIsDAG);
            const std::string assetClass = obj->GetAssetClass();
            AssertMsgFmt(!assetClass.empty(), "Error: asset '%s' has no defined class", obj->GetAssetClass());
            childNode.AddValue("class", assetClass);

            // Deleted objects don't need their full attribute list serialised
            if (!(flags & kEnqueueIdOnly))
            {
                obj->Serialise(childNode, kSerialiseExposedOnly);
            }
        };

        Json::Node node = m_outboundCmdQueue->Create(eventId);
        if (flags & kEnqueueAll)
        {
            for (auto& obj : m_sceneContainer->GenericObjects()) { SerialiseImpl(node, obj); }
        }
        else if (flags & kEnqueueSelected)
        {
            for (auto& obj : m_selectionCtx.selectedObjects) { SerialiseImpl(node, obj); }
        }
        else if (flags & kEnqueueOne)
        {
            SerialiseImpl(node, asset);
        }

        m_outboundCmdQueue->Enqueue();  // Enqueue the staged command
    }

    __host__ uint Host::GI2DRenderer::OnToggleRun(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        m_isRunning = !m_isRunning;
        Log::Warning(m_isRunning ? "Running" : "Paused");

        m_uiGraph.SetState("kIdleState");
        return kUIStateOkay;
    }

    __host__ uint Host::GI2DRenderer::OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        Log::Success("Back home!");
        return kUIStateOkay;
    }

    __host__ uint Host::GI2DRenderer::OnDeleteSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        if (m_selectionCtx.selectedObjects.empty()) { return kUIStateOkay; }

        //std::lock_guard <std::mutex> lock(m_resourceMutex);

        Host::SceneObjectContainer& sceneObjects = m_sceneContainer->SceneObjects();
        int emptyIdx = -1;
        int numDeleted = 0;
        for (int primIdx = 0; primIdx < sceneObjects.Size(); ++primIdx)
        {
            if (sceneObjects[primIdx]->IsSelected())
            {
                // Erase the object from the container. 
                m_sceneBuilder->EnqueueDeleteObject(sceneObjects[primIdx]->GetAssetID());

                ++numDeleted;
                if (emptyIdx == -1) { emptyIdx = primIdx; }
            }
            else if (emptyIdx >= 0)
            {
                sceneObjects[emptyIdx++] = sceneObjects[primIdx];
            }
        }

        Assert(numDeleted <= sceneObjects.Size());
        sceneObjects.Resize(sceneObjects.Size() - numDeleted);
        Log::Error("Delete!");

        EnqueueOutboundSerialisation("OnDeleteObject", kEnqueueSelected | kEnqueueIdOnly);

        // Clear the selected object list
        m_selectionCtx.selectedObjects.clear();

        SignalDirty(kDirtyObjectExistence);

        return kUIStateOkay;
    }

    __host__ uint Host::GI2DRenderer::OnMoveSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        //std::lock_guard <std::mutex> lock(m_resourceMutex);
        
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kMoveSceneObjectBegin")
        {
            m_selectionCtx.dragAnchor = m_viewCtx.mousePos;
        }
        /*else if (stateID == "kMoveSceneObjectDragging")
        {
            // Update the selection overlay
            m_selectionCtx.selectedBBox += m_viewCtx.mousePos - m_selectionCtx.dragAnchor;
            m_selectionCtx.dragAnchor = m_viewCtx.mousePos
            SignalDirty(kDirtyUIOverlay);
        }*/

        // Notify the scene objects of the move operation  
        uint objectDirtyFlags = 0u;
        for (auto& obj : m_selectionCtx.selectedObjects)
        {
            Assert(obj->IsSelected());

            // Moving objects 
            obj->OnMove(stateID, m_viewCtx, m_selectionCtx);
        }

        if (stateID == "kMoveSceneObjectDragging")
        {
            // Update the selection overlay
            UpdateSelectedBBox();
            SignalDirty(kDirtyUIOverlay);
        }

        // Enqueue the list of selected scene objects
        EnqueueOutboundSerialisation("OnUpdateObject", kEnqueueSelected);

        return kUIStateOkay;
    }

    __host__ void Host::GI2DRenderer::DeselectAll()
    {
        //std::lock_guard <std::mutex> lock(m_resourceMutex);

        for (auto obj : m_sceneContainer->SceneObjects())
        {
            obj->OnSelect(false);
        }

        m_selectionCtx.selectedObjects.clear();

        SignalDirty(kDirtyUIOverlay);
    }

    __host__ std::string Host::GI2DRenderer::DecideOnClickState(const uint& sourceStateIdx)
    {
        //std::lock_guard <std::mutex> lock(m_resourceMutex);

        // Before deciding whether to lasso or move, test if the mouse has precision-clicked an object. If it has, select it.
        if (m_sceneContainer->SceneBIH().IsConstructed())
        {
            auto& sceneObjects = m_sceneContainer->SceneObjects();
            int hitIdx = -1;
            uint hitResult = kSceneObjectInvalidSelect;
            auto onContainsPrim = [&, this](const uint* primRange) -> bool
            {
                for (int primIdx = primRange[0]; primIdx < primRange[1]; ++primIdx)
                {
                    if (primIdx >= m_sceneContainer->SceneObjects().Size())
                    {
                        int size = m_sceneContainer->SceneObjects().Size();
                        Log::Error("%i, %i", primIdx, size);
                    }
                    if (sceneObjects[primIdx]->GetWorldSpaceBoundingBox().Contains(m_viewCtx.mousePos))
                    {
                        hitResult = sceneObjects[primIdx]->OnMouseClick(m_viewCtx);
                        if (hitResult != kSceneObjectInvalidSelect)
                        {
                            hitIdx = primIdx;
                            return true;
                        }
                    }
                }
                return false;
            };
            m_sceneContainer->SceneBIH().TestPoint(m_viewCtx.mousePos, onContainsPrim);

            // If we've intersected something...
            if (hitIdx != -1)
            {
                // Precision dragging instantaneously selects the object and goes into the object move state
                if (hitResult == kSceneObjectPrecisionDrag)
                {
                    DeselectAll();

                    m_selectionCtx.selectedObjects.push_back(sceneObjects[hitIdx]);
                    m_selectionCtx.selectedObjects.back()->OnSelect(true);

                    m_selectionCtx.isLassoing = false;
                    m_selectionCtx.selectedBBox = sceneObjects[hitIdx]->GetWorldSpaceBoundingBox();

                    return "kMoveSceneObjectBegin";
                }
                // Otherwise, start delegating mouse movements directly to the scene object until the button is lifted
                else if (hitResult == kSceneObjectDelegatedAction)
                {
                    m_delegatedObject = sceneObjects[hitIdx];
                    return "kDelegateSceneObjectBegin";
                }
                else
                {
                    AssertMsgFmt(false, "Invalid hit result: %i", hitResult);
                }
            }
        }

        // If there are no paths selected, enter selection state. Otherwise, enter moving state.
        if (m_selectionCtx.selectedObjects.empty())
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

    __host__ uint Host::GI2DRenderer::OnDelegateSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        //std::lock_guard <std::mutex> lock(m_resourceMutex);
        
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);

        Assert(m_delegatedObject);
        m_delegatedObject->OnDelegateAction(stateID, keyMap, m_viewCtx);

        if (stateID == "kDelegateSceneObjectEnd")
        {
            m_delegatedObject = nullptr;
        }

        return kUIStateOkay;
    }

    __host__ void Host::GI2DRenderer::UpdateSelectedBBox()
    {
        m_selectionCtx.selectedBBox.MakeInvalid();
        for (auto& object : m_selectionCtx.selectedObjects)
        {
            m_selectionCtx.selectedBBox = Union(m_selectionCtx.selectedBBox, object->GetWorldSpaceBoundingBox());
        }
    }

    __host__ uint Host::GI2DRenderer::OnSelectSceneObjects(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kSelectSceneObjectDragging")
        {
            auto& sceneObjects = m_sceneContainer->SceneObjects();
            const bool wasLassoing = m_selectionCtx.isLassoing;
            const int lastNumSelected = m_selectionCtx.selectedObjects.size();

            if (!m_selectionCtx.isLassoing)
            {
                // Deselect all the path segments
                DeselectAll();

                m_selectionCtx.mouseBBox = BBox2f(m_viewCtx.mousePos);
                m_selectionCtx.isLassoing = true;
            }

            m_selectionCtx.mouseBBox.upper = m_viewCtx.mousePos;
            m_selectionCtx.lassoBBox = Grow(Rectify(m_selectionCtx.mouseBBox), m_viewCtx.dPdXY * 2.);
            m_selectionCtx.selectedBBox = BBox2f::Invalid();
            m_selectionCtx.selectedObjects.clear();

            //std::lock_guard <std::mutex> lock(m_resourceMutex);
            if (m_sceneContainer->SceneBIH().IsConstructed())
            {
                int numSelected = 0;
                auto onIntersectPrim = [&sceneObjects, this](const uint* primRange, const bool isInnerNode)
                {
                    // Inner nodes are tested when the bounding box envelops them completely. Hence, there's no need to do a bbox checks.
                    if (isInnerNode)
                    {
                        for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                        {
                            m_selectionCtx.selectedObjects.emplace_back(sceneObjects[idx]);
                            sceneObjects[idx]->OnSelect(true);
                        }
                    }
                    else
                    {
                        for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                        {
                            const auto& bBoxWorld = sceneObjects[idx]->GetWorldSpaceBoundingBox();
                            const bool isCaptured = m_selectionCtx.lassoBBox.Contains(bBoxWorld);
                            if (isCaptured)
                            {
                                m_selectionCtx.selectedObjects.emplace_back(sceneObjects[idx]);
                                m_selectionCtx.selectedBBox = Union(m_selectionCtx.selectedBBox, bBoxWorld);
                            }
                            sceneObjects[idx]->OnSelect(isCaptured);
                        }
                    }
                };
                m_sceneContainer->SceneBIH().TestBBox(m_selectionCtx.lassoBBox, onIntersectPrim);

                // Only if the number of selected primitives has changed
                if (lastNumSelected != m_selectionCtx.selectedObjects.size())
                {
                    if (!m_selectionCtx.selectedObjects.empty() > 0 && !wasLassoing)
                    {
                        m_selectionCtx.isLassoing = false;
                        m_uiGraph.SetState("kMoveSceneObjectBegin");
                    }
                }
            }

            SignalDirty(kDirtyUIOverlay);
            //Log::Success("Selecting!");
        }
        else if (stateID == "kSelectSceneObjectEnd")
        {
            m_selectionCtx.isLassoing = false;
            SignalDirty(kDirtyUIOverlay);

            //Log::Success("Finished!");
        }
        else if (stateID == "kDeselectSceneObject")
        {
            DeselectAll();
            SignalDirty(kDirtyUIOverlay);

            //Log::Success("Finished!");
        }
        else
        {
            return kUIStateError;
        }

        return kUIStateOkay;
    }

    __host__ uint Host::GI2DRenderer::OnCreateSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& trigger)
    {
        //std::lock_guard <std::mutex> lock(m_resourceMutex);

        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kCreateSceneObjectOpen")
        {
            // Try and instantiate the objerct         
            auto newObject = m_sceneObjectFactory.Instantiate(trigger.HashOf(), m_sceneContainer->GenericObjects(), *this, m_sceneContainer);
            m_onCreate.newObject = newObject.DynamicCast<Host::SceneObject>();

            SignalDirty(kDirtyObjectExistence);

            // Emplace the new object with the scene builder ready for integration into the scene
            //m_sceneBuilder->EnqueueEmplaceObject(m_onCreate.newObject);
        }

        // Invoke the event handler of the new object
        Assert(m_onCreate.newObject);
        m_onCreate.newObject->OnCreate(stateID, m_viewCtx);

        // Some objects will automatically finalise themselves. If this happens, we're done.
        if (m_onCreate.newObject->IsFinalised())
        {
            EnqueueOutboundSerialisation("OnCreateObject", kEnqueueOne, m_onCreate.newObject);
            m_uiGraph.SetState("kIdleState");
            return kUIStateOkay;
        }

        if (stateID == "kCreateSceneObjectClose")
        {
            FinaliseNewSceneObject();
        }

        return kUIStateOkay;
    }

    __host__ void Host::GI2DRenderer::FinaliseNewSceneObject()
    {
        Assert(m_onCreate.newObject);

        // If the new object has closed but has not been finalised, delete i
        if (!m_onCreate.newObject->IsFinalised())
        {
            m_sceneBuilder->EnqueueDeleteObject(m_onCreate.newObject->GetAssetID());
            Log::Success("Destroying unfinalised scene object '%s'", m_onCreate.newObject->GetAssetID());
        }
        else
        {
            // Serialise the new object to the outbound queue
            EnqueueOutboundSerialisation("OnCreateObject", kEnqueueOne, m_onCreate.newObject);
        } 

        m_onCreate.newObject = nullptr;
    }

    __host__ void Host::GI2DRenderer::OnCommandsWaiting(CommandQueue& inbound)
    {
        m_commandManager.Flush(inbound, true);
    }

    __host__ void Host::GI2DRenderer::OnRender()
    {        
        //m_resourceMutex.lock();       

        // Flush any keyboard and mouse inputs that have accumulated between now and the beginning of the last frame
        FlushUIEventQueue();

        // Rebuild the scene 
        m_sceneBuilder->Rebuild(true);

        // Prepare the scene objects
        m_sceneContainer->Prepare();
        m_overlayRenderer->Prepare(m_viewCtx, m_selectionCtx);
            
        /*if (m_renderTimer.Get() > 0.1f)
        {
            for (auto& camera : m_sceneContainer->Cameras())
            {
                if (camera->Prepare())
                {
                    camera->Integrate();
                }
            }
            //m_renderTimer.Reset();
            //Log::Write("-----");
        }*/        
  
        m_overlayRenderer->Render(); 

        // If a blit is in progress, skip the composite step entirely.
        // TODO: Make this respond intelligently to frame rate. If the CUDA renderer is running at a lower FPS than the D3D renderer then it should wait rather than
        // than skipping frames like this.
        //m_renderSemaphore.Wait(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress);
        if (!m_renderSemaphore.Try(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress, false)) { return; }

        // Composite the render layers
        m_compositeImage->Clear(vec4(kZero, 1.0f));
        m_overlayRenderer->Composite(m_compositeImage);

        m_renderSemaphore.Try(kRenderManagerCompInProgress, kRenderManagerCompFinished, true);

        // Clean all the objects in the scene container ready for the next iteration
        m_sceneContainer->Clean();
        Clean();
    }

    __host__ void Host::GI2DRenderer::OnMouseButton(const uint code, const bool isDown)
    {
        // Is the view being changed? 
        if (isDown && (code == VK_MBUTTON || code == VK_RBUTTON || IsKeyDown(VK_SHIFT)))
        {
            m_viewCtx.dragAnchor = vec2(m_mouse.pos);
            m_viewCtx.rotAxis = normalize(m_viewCtx.dragAnchor - vec2(m_clientWidth, m_clientHeight) * 0.5f);
            m_viewCtx.transAnchor = m_viewCtx.transform.trans;
            m_viewCtx.scaleAnchor = m_viewCtx.transform.scale;
            m_viewCtx.rotAnchor = m_viewCtx.transform.rotate;
        }
    }

    __host__ void Host::GI2DRenderer::OnMouseMove()
    {
        OnViewChange();

        {
            //std::lock_guard <std::mutex> lock(m_resourceMutex);
            m_viewCtx.mousePos = m_viewCtx.transform.matrix * vec2(m_mouse.pos);
        }
    }

    __host__ void Host::GI2DRenderer::OnViewChange()
    {
        auto& transform = m_viewCtx.transform;

        // Zooming?
        if (IsMouseButtonDown(VK_RBUTTON))
        {
            float logScaleAnchor = std::log2(std::max(1e-10f, m_viewCtx.scaleAnchor));
            logScaleAnchor += m_viewCtx.zoomSpeed * float(m_mouse.pos.y - m_viewCtx.dragAnchor.y) / m_clientHeight;
            transform.scale = std::pow(2.0, logScaleAnchor);

            //Log::Write("Scale: %f", transform.scale);
        }
        // Rotating?
        else if (IsKeyDown(VK_SHIFT) && IsMouseButtonDown(VK_LBUTTON))
        {
            const vec2 delta = normalize(vec2(m_mouse.pos) - vec2(m_clientWidth, m_clientHeight) * 0.5f);
            const float theta = std::acos(dot(delta, m_viewCtx.rotAxis)) * (float(dot(delta, vec2(m_viewCtx.rotAxis.y, -m_viewCtx.rotAxis.x)) < 0.0f) * 2.0 - 1.0f);
            transform.rotate = m_viewCtx.rotAnchor + theta;

            if (std::abs(std::fmod(transform.rotate, kHalfPi)) < 0.05f) { transform.rotate = std::round(transform.rotate / kHalfPi) * kHalfPi; }

            //Log::Write("Theta: %f", transform.rotate);
        }
        // Translating
        else if(IsMouseButtonDown(VK_MBUTTON))
        {
            // Update the transformation
            const mat3 newMat = ConstructViewMatrix(m_viewCtx.transAnchor, transform.rotate, transform.scale) * m_clientToNormMatrix;
            const vec2 dragDelta = (newMat * vec2(m_viewCtx.dragAnchor)) - (newMat * vec2(m_mouse.pos));
            transform.trans = m_viewCtx.transAnchor + dragDelta;

            //Log::Write("Trans: %s", m_viewCtx.trans.format());
        }

        // Update the parameters in the overlay renderer
        {
            //std::lock_guard <std::mutex> lock(m_resourceMutex);
            transform.matrix = ConstructViewMatrix(transform.trans, transform.rotate, transform.scale) * m_clientToNormMatrix;
            m_viewCtx.Prepare();

            // Mark the scene as dirty
            //SignalDirty(kDirtyUIOverlay);
        }
    }

    __host__ void Host::GI2DRenderer::OnResizeClient()
    {
    } 

    __host__ void Host::GI2DRenderer::OnFocusChange(const bool isSet)
    {
        // Finalise any objects that are in the process of being created
        /*if (m_onCreate.newObject)
        {
            FinaliseNewSceneObject();
        }*/
    }

    __host__ bool Host::GI2DRenderer::Serialise(Json::Document& json, const int flags)
    {
        return true;
    }
}