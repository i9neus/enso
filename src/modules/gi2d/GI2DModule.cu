#include "GI2DModule.cuh"

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

    __host__ Host::GI2DModule::GI2DModule(const InitCtx& initCtx, std::shared_ptr<CommandQueue> outQueue) :
        Dirtyable(initCtx),
        ModuleBase(outQueue),
        m_isRunning(true)
    {
        // Load the object schema
        SerialisableObjectSchemaContainer::Load("GI2DSchema.json");

        // Register the outbound commands
        m_outboundCmdQueue->RegisterCommand("OnCreateObject");
        m_outboundCmdQueue->RegisterCommand("OnUpdateObject");
        m_outboundCmdQueue->RegisterCommand("OnDeleteObject");

        // Register the inbound command handlers
        m_commandManager.RegisterEventHandler("OnUpdateObject", this, &Host::GI2DModule::OnInboundUpdateObject);

        // Register the scene object instantiators
        RegisterInstantiators();

        // Declare the transition graph that will drive the UI
        DeclareStateTransitionGraph();

        // Declare any listeners that will respond to changes in the scene graph
        DeclareListeners();
    }

    __host__ Host::GI2DModule::~GI2DModule() noexcept
    {
        m_overlayRenderer.DestroyAsset();
        m_voxelProxyGrid.DestroyAsset();

        m_sceneBuilder.DestroyAsset();
        m_sceneContainer.DestroyAsset();
    }

    __host__ void Host::GI2DModule::RegisterInstantiators()
    {
        m_sceneObjectFactory.RegisterInstantiator<Host::LineStrip>(VirtualKeyMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());
        m_sceneObjectFactory.RegisterInstantiator<Host::KIFS>(VirtualKeyMap({ {'W', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());
        m_sceneObjectFactory.RegisterInstantiator<Host::PerspectiveCamera>(VirtualKeyMap({ {'E', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());
        m_sceneObjectFactory.RegisterInstantiator<Host::OmniLight>(VirtualKeyMap({ {'A', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }).HashOf());

        //m_sceneObjectFactory.RegisterInstantiator<Host::OverlayLayer>();
        //m_sceneObjectFactory.RegisterInstantiator<Host::VoxelProxyGrid>();
    }

    __host__ void Host::GI2DModule::DeclareStateTransitionGraph()
    {
        m_uiGraph.DeclareState("kIdleState", this, &Host::GI2DModule::OnIdleState);

        // Create scene object
        m_uiGraph.DeclareState("kCreateDrawableObjectOpen", this, &GI2DModule::OnCreateDrawableObject);
        m_uiGraph.DeclareState("kCreateDrawableObjectHover", this, &GI2DModule::OnCreateDrawableObject);
        m_uiGraph.DeclareState("kCreateDrawableObjectAppend", this, &GI2DModule::OnCreateDrawableObject);
        m_uiGraph.DeclareState("kCreateDrawableObjectClose", this, &GI2DModule::OnCreateDrawableObject);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateDrawableObjectOpen", VirtualKeyMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateDrawableObjectOpen", VirtualKeyMap({ {'A', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateDrawableObjectOpen", VirtualKeyMap({ {'E', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateDrawableObjectOpen", VirtualKeyMap({ {'W', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kCreateDrawableObjectOpen", "kCreateDrawableObjectHover");
        m_uiGraph.DeclareDeterministicTransition("kCreateDrawableObjectHover", "kCreateDrawableObjectHover", VirtualKeyMap::Nothing(), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kCreateDrawableObjectHover", "kCreateDrawableObjectAppend", VirtualKeyMap(VK_LBUTTON, kOnButtonDepressed), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kCreateDrawableObjectAppend", "kCreateDrawableObjectHover");
        m_uiGraph.DeclareDeterministicTransition("kCreateDrawableObjectHover", "kCreateDrawableObjectClose", VirtualKeyMap(VK_RBUTTON, kOnButtonDepressed), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kCreateDrawableObjectClose", "kIdleState");

        // Delegate mouse actions to scene objects
        m_uiGraph.DeclareState("kDelegateDrawableObjectBegin", this, &GI2DModule::OnDelegateDrawableObject);
        m_uiGraph.DeclareState("kDelegateDrawableObjectEnd", this, &GI2DModule::OnDelegateDrawableObject);
        m_uiGraph.DeclareState("kDelegateDrawableObjectDragging", this, &GI2DModule::OnDelegateDrawableObject);
        m_uiGraph.DeclareDeterministicAutoTransition("kDelegateDrawableObjectBegin", "kDelegateDrawableObjectDragging");
        m_uiGraph.DeclareDeterministicTransition("kDelegateDrawableObjectDragging", "kDelegateDrawableObjectDragging", VirtualKeyMap(VK_LBUTTON, kButtonDown), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kDelegateDrawableObjectDragging", "kDelegateDrawableObjectEnd", VirtualKeyMap(VK_LBUTTON, kOnButtonReleased), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kDelegateDrawableObjectEnd", "kIdleState");

        // Select/deselect scene object
        m_uiGraph.DeclareState("kSelectDrawableObjectDragging", this, &GI2DModule::OnSelectDrawableObjects);
        m_uiGraph.DeclareState("kSelectDrawableObjectEnd", this, &GI2DModule::OnSelectDrawableObjects);
        m_uiGraph.DeclareState("kDeselectDrawableObject", this, &GI2DModule::OnSelectDrawableObjects);
        m_uiGraph.DeclareNonDeterministicTransition("kIdleState", VirtualKeyMap(VK_LBUTTON, kOnButtonDepressed), 0, this, &GI2DModule::DecideOnClickState);
        m_uiGraph.DeclareDeterministicTransition("kSelectDrawableObjectDragging", "kSelectDrawableObjectDragging", VirtualKeyMap(VK_LBUTTON, kButtonDown), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kSelectDrawableObjectDragging", "kSelectDrawableObjectEnd", VirtualKeyMap(VK_LBUTTON, kOnButtonReleased), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kSelectDrawableObjectEnd", "kIdleState");
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeselectDrawableObject", VirtualKeyMap(VK_RBUTTON, kOnButtonDepressed), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kDeselectDrawableObject", "kIdleState");

        // Move scene object
        m_uiGraph.DeclareState("kMoveDrawableObjectBegin", this, &GI2DModule::OnMoveDrawableObject);
        m_uiGraph.DeclareState("kMoveDrawableObjectDragging", this, &GI2DModule::OnMoveDrawableObject);
        m_uiGraph.DeclareState("kMoveDrawableObjectEnd", this, &GI2DModule::OnMoveDrawableObject);
        //m_uiGraph.DeclareNonDeterministicTransition("kIdleState", nullptr, MouseButtonMap(VK_LBUTTON, kOnButtonDepressed), 0, this, &GI2DModule::DecideOnClickState);
        m_uiGraph.DeclareDeterministicAutoTransition("kMoveDrawableObjectBegin", "kMoveDrawableObjectDragging");
        m_uiGraph.DeclareDeterministicTransition("kMoveDrawableObjectDragging", "kMoveDrawableObjectDragging", VirtualKeyMap(VK_LBUTTON, kButtonDown), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kMoveDrawableObjectDragging", "kMoveDrawableObjectEnd", VirtualKeyMap(VK_LBUTTON, kOnButtonReleased), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kMoveDrawableObjectEnd", "kIdleState");

        // Delete scene object
        m_uiGraph.DeclareState("kDeleteDrawableObjects", this, &GI2DModule::OnDeleteDrawableObject);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeleteDrawableObjects", VirtualKeyMap({ {VK_DELETE, kOnButtonDepressed} }), 0);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeleteDrawableObjects", VirtualKeyMap({ {VK_BACK, kOnButtonDepressed} }), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kDeleteDrawableObjects", "kIdleState");

        // Utils
        m_uiGraph.DeclareState("kToggleRun", this, &GI2DModule::OnToggleRun);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kToggleRun", VirtualKeyMap({ {VK_SPACE, kOnButtonDepressed} }), 0);

        m_uiGraph.Finalise();
    }

    __host__ void Host::GI2DModule::DeclareListeners()
    {
        Listen({ kDirtyViewportObjectBBox });
    }

    __host__ void Host::GI2DModule::OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller)
    {
        SetDirty(flag);
    }

    std::shared_ptr<ModuleBase> Host::GI2DModule::Instantiate(std::shared_ptr<CommandQueue> outQueue)
    {
        AssetHandle<ModuleBase> newAsset = AssetAllocator::CreateAsset<Host::GI2DModule>("gi2d", outQueue);
        return std::shared_ptr<ModuleBase>(newAsset);
    }

    __host__ void Host::GI2DModule::OnInitialise()
    {
        m_viewCtx.transform = ViewTransform2D(m_clientToNormMatrix, vec2(0.f), 0.f, 1.0f);
        m_viewCtx.dPdXY = length(vec2(m_viewCtx.transform.matrix.i00, m_viewCtx.transform.matrix.i10));
        m_viewCtx.zoomSpeed = 10.0f;
        m_viewCtx.sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));

        LoadScene();

        m_sceneBuilder->Rebuild(true);
    }

    __host__ void Host::GI2DModule::LoadScene()
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

    __host__ void Host::GI2DModule::OnInboundUpdateObject(const Json::Node& node)
    {
        Assert(node.IsArray(), "Expected array");
        for(int idx = 0; idx < node.Size(); ++idx)
        {
            Json::Node itemNode = node[idx];
            Assert(itemNode.IsObject());

            std::string id;
            itemNode.GetValue("id", id, Json::kRequiredAssert | Json::kNotBlank);
            auto objectHandle = m_sceneContainer->GenericObjects().FindByID(id);

            if (!objectHandle)
            {
                Log::Warning("Error: '%s' is not a valid scene object.", id);
                continue;
            }

            objectHandle->Deserialise(itemNode, Json::kSilent);
        }
    }

    __host__ void Host::GI2DModule::EnqueueOutboundSerialisation(const std::string& eventId, const int flags, const AssetHandle<Host::GenericObject> asset)
    {
        // TODO: This should be smarter and happen automatically whenever an object is dirtied. 
        
        if (!m_outboundCmdQueue->IsRegistered(eventId)) { return; }

        // Lambda to do the actual serialisation
        auto SerialiseImpl = [&](Json::Node& node, const AssetHandle<Host::GenericObject>& obj) -> void
        {           
            // Create a new child object and add its class ID for the schema
            const std::string assetClass = obj->GetAssetClass();
            AssertMsgFmt(!assetClass.empty(), "Error: asset '%s' has no defined class", obj->GetAssetClass());
            node.AddValue("id", obj->GetAssetID());
            //node.AddValue("dagPath", obj->GetAssetDAGPath());
            node.AddValue("class", assetClass);

            // Deleted objects don't need their full attribute list serialised
            if (!(flags & kEnqueueIdOnly))
            {
                obj->Serialise(node, kSerialiseExposedOnly);
            }
        };

        Json::Node nodeArray = m_outboundCmdQueue->Create(eventId).MakeArray();

        if (flags & kEnqueueAll)
        {
            for (auto& obj : m_sceneContainer->GenericObjects()) { SerialiseImpl(nodeArray.AppendArrayObject(), obj); }
        }
        else if (flags & kEnqueueSelected)
        {
            for (auto& obj : m_selectionCtx.selectedObjects) { SerialiseImpl(nodeArray.AppendArrayObject(), obj); }
        }
        else if (flags & kEnqueueOne)
        {
            SerialiseImpl(nodeArray.AppendArrayObject(), asset);
        }

        Log::Warning(nodeArray.Stringify(true));

        m_outboundCmdQueue->Enqueue();  // Enqueue the staged command
    }

    __host__ uint Host::GI2DModule::OnToggleRun(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        m_isRunning = !m_isRunning;
        Log::Warning(m_isRunning ? "Running" : "Paused");

        m_uiGraph.SetState("kIdleState");
        return kUIStateOkay;
    }

    __host__ uint Host::GI2DModule::OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        Log::Success("Back home!");
        return kUIStateOkay;
    }

    __host__ uint Host::GI2DModule::OnDeleteDrawableObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        if (m_selectionCtx.selectedObjects.empty()) { return kUIStateOkay; }        

        Host::DrawableObjectContainer& sceneObjects = m_sceneContainer->DrawableObjects();
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

    __host__ uint Host::GI2DModule::OnMoveDrawableObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kMoveDrawableObjectBegin")
        {
            m_selectionCtx.dragAnchor = m_viewCtx.mousePos;
        }
        /*else if (stateID == "kMoveDrawableObjectDragging")
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

        if (stateID == "kMoveDrawableObjectDragging")
        {

        }

        // Enqueue the list of selected scene objects
        EnqueueOutboundSerialisation("OnUpdateObject", kEnqueueSelected);

        return kUIStateOkay;
    }

    __host__ void Host::GI2DModule::DeselectAll()
    {
        for (auto obj : m_sceneContainer->DrawableObjects())
        {
            obj->OnSelect(false);
        }

        m_selectionCtx.selectedObjects.clear();

        SignalDirty(kDirtyUIOverlay);
    }

    __host__ std::string Host::GI2DModule::DecideOnClickState(const uint& sourceStateIdx)
    {
        // Before deciding whether to lasso or move, test if the mouse has precision-clicked an object. If it has, select it.
        if (m_sceneContainer->SceneBIH().IsConstructed())
        {
            auto& sceneObjects = m_sceneContainer->DrawableObjects();
            constexpr int kInvalidHit = -1;
            int hitIdx = kInvalidHit;
            uint hitResult = kDrawableObjectInvalidSelect;
            auto onContainsPrim = [&, this](const uint* primRange, const uint* primIdxs) -> bool
            {
                for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                {
                    const uint primIdx = primIdxs[idx];
                    if (primIdx >= m_sceneContainer->DrawableObjects().Size())
                    {
                        int size = m_sceneContainer->DrawableObjects().Size();
                        Log::Error("%i, %i", primIdx, size);
                    }

                    if (sceneObjects[primIdx]->GetWorldSpaceBoundingBox().Contains(m_viewCtx.mousePos))
                    {
                        hitResult = sceneObjects[primIdx]->OnMouseClick(m_viewCtx);
                        if (hitResult != kDrawableObjectInvalidSelect)
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
            if (hitIdx != kInvalidHit)
            {
                // Precision dragging instantaneously selects the object and goes into the object move state
                if (hitResult == kDrawableObjectPrecisionDrag)
                {
                    DeselectAll();

                    m_selectionCtx.selectedObjects.push_back(sceneObjects[hitIdx]);
                    m_selectionCtx.selectedObjects.back()->OnSelect(true);

                    m_selectionCtx.isLassoing = false;
                    m_selectionCtx.selectedBBox = sceneObjects[hitIdx]->GetWorldSpaceBoundingBox();

                    return "kMoveDrawableObjectBegin";
                }
                // Otherwise, start delegating mouse movements directly to the scene object until the button is lifted
                else if (hitResult == kDrawableObjectDelegatedAction)
                {
                    m_delegatedObject = sceneObjects[hitIdx];
                    return "kDelegateDrawableObjectBegin";
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
            return "kSelectDrawableObjectDragging";
        }
        else
        {
            //Assert(selection.selectedBBox.HasValidArea());
            if (Grow(m_selectionCtx.selectedBBox, m_viewCtx.dPdXY * 2.f).Contains(m_viewCtx.mousePos))
            {
                return "kMoveDrawableObjectBegin";
            }
            else
            {
                // Deselect everything
                DeselectAll();
                return "kSelectDrawableObjectDragging";
            }
        }
        return "kSelectDrawableObjectDragging";
    }

    __host__ uint Host::GI2DModule::OnDelegateDrawableObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);

        Assert(m_delegatedObject);
        m_delegatedObject->OnDelegateAction(stateID, keyMap, m_viewCtx);

        if (stateID == "kDelegateDrawableObjectEnd")
        {
            m_delegatedObject = nullptr;
        }

        return kUIStateOkay;
    }

    __host__ void Host::GI2DModule::UpdateSelectedBBox()
    {
        m_selectionCtx.selectedBBox.MakeInvalid();
        for (auto& object : m_selectionCtx.selectedObjects)
        {
            m_selectionCtx.selectedBBox = Union(m_selectionCtx.selectedBBox, object->GetWorldSpaceBoundingBox());
        }
        SignalDirty(kDirtyUIOverlay);
    }

    __host__ uint Host::GI2DModule::OnSelectDrawableObjects(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kSelectDrawableObjectDragging")
        {
            auto& sceneObjects = m_sceneContainer->DrawableObjects();
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

            if (m_sceneContainer->SceneBIH().IsConstructed())
            {
                int numSelected = 0;
                auto onIntersectPrim = [&sceneObjects, this](const uint* primRange, const uint* primIdxs, const bool isInnerNode)
                {
                    // Inner nodes are tested when the bounding box envelops them completely. Hence, there's no need to do a bbox checks.
                    if (isInnerNode)
                    {
                        for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                        {
                            const uint primIdx = primIdxs[idx];
                            m_selectionCtx.selectedObjects.emplace_back(sceneObjects[primIdx]);
                            sceneObjects[primIdx]->OnSelect(true);
                        }
                    }
                    else
                    {
                        for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                        {
                            const uint primIdx = primIdxs[idx];
                            const auto& bBoxWorld = sceneObjects[primIdx]->GetWorldSpaceBoundingBox();
                            const bool isCaptured = m_selectionCtx.lassoBBox.Contains(bBoxWorld);
                            if (isCaptured)
                            {
                                Log::Debug("Selected %i", primIdx);
                                m_selectionCtx.selectedObjects.emplace_back(sceneObjects[primIdx]);
                                m_selectionCtx.selectedBBox = Union(m_selectionCtx.selectedBBox, bBoxWorld);
                            }
                            sceneObjects[primIdx]->OnSelect(isCaptured);
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
                        m_uiGraph.SetState("kMoveDrawableObjectBegin");
                    }
                }
            }

            SignalDirty(kDirtyUIOverlay);
            //Log::Success("Selecting!");
        }
        else if (stateID == "kSelectDrawableObjectEnd")
        {
            m_selectionCtx.isLassoing = false;
            SignalDirty(kDirtyUIOverlay);

            Log::Debug("Selected:");
            for (const auto& obj : m_selectionCtx.selectedObjects)
            {
                Log::Debug("  - %s", obj->GetAssetID());
            }

            //Log::Success("Finished!");
        }
        else if (stateID == "kDeselectDrawableObject")
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

    __host__ uint Host::GI2DModule::OnCreateDrawableObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& trigger)
    {
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kCreateDrawableObjectOpen")
        {
            // Try and instantiate the objerct         
            auto newObject = m_sceneObjectFactory.Instantiate(trigger.HashOf(), m_sceneContainer->GenericObjects(), *this, m_sceneContainer);
            m_onCreate.newObject = newObject.DynamicCast<Host::DrawableObject>();

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

        if (stateID == "kCreateDrawableObjectClose")
        {
            FinaliseNewDrawableObject();
        }

        return kUIStateOkay;
    }

    __host__ void Host::GI2DModule::FinaliseNewDrawableObject()
    {
        Assert(m_onCreate.newObject);

        // If the new object has closed but has not been finalised, delete it
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

    __host__ void Host::GI2DModule::OnCommandsWaiting(CommandQueue& inbound)
    {
        m_commandManager.Flush(inbound, false);
    }

    __host__ void Host::GI2DModule::OnRender()
    {        
        // Flush any keyboard and mouse inputs that have accumulated between now and the beginning of the last frame
        FlushUIEventQueue();

        // Update the overlay if the scene is dirty
        const bool updateSelection = !m_sceneBuilder->IsClean();

        // Rebuild the scene 
        m_sceneBuilder->Rebuild(false);

        if (!updateSelection)
        {
            UpdateSelectedBBox();
        }

        // Prepare the scene objects
        //m_sceneContainer->Prepare();
        m_overlayRenderer->Prepare(m_viewCtx, m_selectionCtx);
            
        //if (m_renderTimer.Get() > 0.1f)
        //{
            for (auto& camera : m_sceneContainer->Cameras())
            {   
                camera->Integrate();
            }
            //m_renderTimer.Reset();
            //Log::Write("-----");
        //}
  
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

    __host__ void Host::GI2DModule::OnMouseButton(const uint code, const bool isDown)
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

    __host__ void Host::GI2DModule::OnMouseMove()
    {
        OnViewChange();

        m_viewCtx.mousePos = m_viewCtx.transform.matrix * vec2(m_mouse.pos);
    }

    __host__ void Host::GI2DModule::OnViewChange()
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
            transform.matrix = ConstructViewMatrix(transform.trans, transform.rotate, transform.scale) * m_clientToNormMatrix;
            m_viewCtx.Prepare();

            // Mark the scene as dirty
            //SignalDirty(kDirtyUIOverlay);
        }
    }

    __host__ void Host::GI2DModule::OnResizeClient()
    {
    } 

    __host__ void Host::GI2DModule::OnFocusChange(const bool isSet)
    {
        // Finalise any objects that are in the process of being created
        /*if (m_onCreate.newObject)
        {
            FinaliseNewDrawableObject();
        }*/
    }

    __host__ bool Host::GI2DModule::Serialise(Json::Document& json, const int flags)
    {
        return true;
    }
}