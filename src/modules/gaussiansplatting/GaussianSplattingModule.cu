#include "GaussianSplattingModule.cuh"

#include "core/math/Math.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/assets/GenericObjectContainer.cuh"
#include "core/containers/Vector.cuh"
#include "core/utils/Tuple.cuh"
#include "viewport/ViewportRenderer.cuh"

#include "core/2d/bih/BIH2DAsset.cuh"

#include "pathtracer/PathTracer.cuh"
#include "splatcloud/SplatRasteriser.cuh"
#include "splatcloud/SplatOptimiser.cuh"

#include "scene/pointclouds/GaussianPointCloud.cuh"
#include "scene/SceneBuilder.cuh"
#include "scene/SceneContainer.cuh"

#include "io/SerialisableObjectSchema.h"

//#include "kernels/gi2d/ObjectDebugger.cuh"

namespace Enso
{

    __host__ Host::GaussianSplattingModule::GaussianSplattingModule(const InitCtx& initCtx, std::shared_ptr<CommandQueue> outQueue) :
        ModuleBase(initCtx, outQueue),
        m_isRunning(true)
    {
        // Load the object schema
        SerialisableObjectSchemaContainer::Load("GaussianSplattingSchema.json");

        // Register the outbound commands
        m_outboundCmdQueue->RegisterCommand("OnCreateObject");
        m_outboundCmdQueue->RegisterCommand("OnUpdateObject");
        m_outboundCmdQueue->RegisterCommand("OnDeleteObject");

        // Register the inbound command handlers
        m_commandManager.RegisterEventHandler("OnUpdateObject", this, &Host::GaussianSplattingModule::OnInboundUpdateObject);

        // Register the scene object instantiators
        RegisterInstantiators();

        // Declare the transition graph that will drive the UI
        DeclareStateTransitionGraph();

        // Declare any listeners that will respond to changes in the scene graph
        DeclareListeners();
    }

    __host__ Host::GaussianSplattingModule::~GaussianSplattingModule() noexcept
    {
        UnloadScene();
    }

    __host__ void Host::GaussianSplattingModule::RegisterInstantiators()
    {
        m_componentFactory.RegisterInstantiator<Host::PathTracer>(VirtualKeyMap({ {'Q', kOnButtonDepressed}, {KEY_CONTROL, kButtonDown} }).HashOf());
        m_componentFactory.RegisterInstantiator<Host::SplatRasteriser>(VirtualKeyMap({ {'W', kOnButtonDepressed}, {KEY_CONTROL, kButtonDown} }).HashOf());


        //m_componentFactory.RegisterInstantiator<Host::ViewportRenderer>();
        //m_componentFactory.RegisterInstantiator<Host::VoxelProxyGrid>();
    }

    __host__ void Host::GaussianSplattingModule::DeclareStateTransitionGraph()
    {
        m_uiGraph.DeclareState("kIdleState", this, &Host::GaussianSplattingModule::OnIdleState);

        // Create scene object
        m_uiGraph.DeclareState("kCreateDrawableObjectOpen", this, &GaussianSplattingModule::OnCreateViewportObject);
        m_uiGraph.DeclareState("kCreateDrawableObjectHover", this, &GaussianSplattingModule::OnCreateViewportObject);
        m_uiGraph.DeclareState("kCreateDrawableObjectAppend", this, &GaussianSplattingModule::OnCreateViewportObject);
        m_uiGraph.DeclareState("kCreateDrawableObjectClose", this, &GaussianSplattingModule::OnCreateViewportObject);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateDrawableObjectOpen", VirtualKeyMap({ {'Q', kOnButtonDepressed}, {KEY_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreateDrawableObjectOpen", VirtualKeyMap({ {'W', kOnButtonDepressed}, {KEY_CONTROL, kButtonDown} }), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kCreateDrawableObjectOpen", "kCreateDrawableObjectHover");
        m_uiGraph.DeclareDeterministicTransition("kCreateDrawableObjectHover", "kCreateDrawableObjectHover", VirtualKeyMap::Nothing(), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kCreateDrawableObjectHover", "kCreateDrawableObjectAppend", VirtualKeyMap(KEY_LBUTTON, kOnButtonDepressed), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kCreateDrawableObjectAppend", "kCreateDrawableObjectHover");
        m_uiGraph.DeclareDeterministicTransition("kCreateDrawableObjectHover", "kCreateDrawableObjectClose", VirtualKeyMap(KEY_RBUTTON, kOnButtonDepressed), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kCreateDrawableObjectClose", "kIdleState");

        // Delegate mouse actions to scene objects
        m_uiGraph.DeclareState("kDelegateDrawableObjectBegin", this, &GaussianSplattingModule::OnDelegateViewportObject);
        m_uiGraph.DeclareState("kDelegateDrawableObjectEnd", this, &GaussianSplattingModule::OnDelegateViewportObject);
        m_uiGraph.DeclareState("kDelegateDrawableObjectDragging", this, &GaussianSplattingModule::OnDelegateViewportObject);
        m_uiGraph.DeclareDeterministicAutoTransition("kDelegateDrawableObjectBegin", "kDelegateDrawableObjectDragging");
        m_uiGraph.DeclareDeterministicTransition("kDelegateDrawableObjectDragging", "kDelegateDrawableObjectDragging", VirtualKeyMap(KEY_LBUTTON, kButtonDown), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kDelegateDrawableObjectDragging", "kDelegateDrawableObjectEnd", VirtualKeyMap(KEY_LBUTTON, kOnButtonReleased), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kDelegateDrawableObjectEnd", "kIdleState");

        // Select/deselect scene object
        m_uiGraph.DeclareState("kSelectDrawableObjectDragging", this, &GaussianSplattingModule::OnSelectViewportObjects);
        m_uiGraph.DeclareState("kSelectDrawableObjectEnd", this, &GaussianSplattingModule::OnSelectViewportObjects);
        m_uiGraph.DeclareState("kDeselectDrawableObject", this, &GaussianSplattingModule::OnSelectViewportObjects);
        m_uiGraph.DeclareNonDeterministicTransition("kIdleState", VirtualKeyMap(KEY_LBUTTON, kOnButtonDepressed), 0, this, &GaussianSplattingModule::DecideOnClickState);
        m_uiGraph.DeclareDeterministicTransition("kSelectDrawableObjectDragging", "kSelectDrawableObjectDragging", VirtualKeyMap(KEY_LBUTTON, kButtonDown), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kSelectDrawableObjectDragging", "kSelectDrawableObjectEnd", VirtualKeyMap(KEY_LBUTTON, kOnButtonReleased), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kSelectDrawableObjectEnd", "kIdleState");
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeselectDrawableObject", VirtualKeyMap(KEY_RBUTTON, kOnButtonDepressed), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kDeselectDrawableObject", "kIdleState");

        // Move scene object
        m_uiGraph.DeclareState("kMoveDrawableObjectBegin", this, &GaussianSplattingModule::OnMoveViewportObject);
        m_uiGraph.DeclareState("kMoveDrawableObjectDragging", this, &GaussianSplattingModule::OnMoveViewportObject);
        m_uiGraph.DeclareState("kMoveDrawableObjectEnd", this, &GaussianSplattingModule::OnMoveViewportObject);
        //m_uiGraph.DeclareNonDeterministicTransition("kIdleState", nullptr, MouseButtonMap(KEY_LBUTTON, kOnButtonDepressed), 0, this, &GaussianSplattingModule::DecideOnClickState);
        m_uiGraph.DeclareDeterministicAutoTransition("kMoveDrawableObjectBegin", "kMoveDrawableObjectDragging");
        m_uiGraph.DeclareDeterministicTransition("kMoveDrawableObjectDragging", "kMoveDrawableObjectDragging", VirtualKeyMap(KEY_LBUTTON, kButtonDown), kUITriggerOnMouseMove);
        m_uiGraph.DeclareDeterministicTransition("kMoveDrawableObjectDragging", "kMoveDrawableObjectEnd", VirtualKeyMap(KEY_LBUTTON, kOnButtonReleased), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kMoveDrawableObjectEnd", "kIdleState");

        // Delete scene object
        m_uiGraph.DeclareState("kDeleteDrawableObjects", this, &GaussianSplattingModule::OnDeleteViewportObject);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeleteDrawableObjects", VirtualKeyMap({ {KEY_DELETE, kOnButtonDepressed} }), 0);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeleteDrawableObjects", VirtualKeyMap({ {KEY_BACK, kOnButtonDepressed} }), 0);
        m_uiGraph.DeclareDeterministicAutoTransition("kDeleteDrawableObjects", "kIdleState");

        // Utils
        m_uiGraph.DeclareState("kToggleRun", this, &GaussianSplattingModule::OnToggleRun);
        m_uiGraph.DeclareDeterministicTransition("kIdleState", "kToggleRun", VirtualKeyMap({ {KEY_SPACE, kOnButtonDepressed} }), 0);

        m_uiGraph.Finalise(); 
    }

    __host__ void Host::GaussianSplattingModule::DeclareListeners()
    {
        Listen({ kDirtyParams });
    }

    __host__ void Host::GaussianSplattingModule::OnDirty(const DirtinessEvent& flag, AssetHandle<Host::Asset>& caller)
    {                
        m_syncObjectParamsSet.emplace(caller->GetAssetID());
    }

    AssetHandle<ModuleBase> Host::GaussianSplattingModule::Instantiate(std::shared_ptr<CommandQueue> outQueue)
    {
        return AssetAllocator::CreateAsset<Host::GaussianSplattingModule>("gaussiansplatting", outQueue);
    }

    __host__ void Host::GaussianSplattingModule::OnInitialise()
    {
        m_viewCtx.transform = ViewTransform2D(m_clientToNormMatrix, vec2(0.f), 0.f, 1.0f);
        m_viewCtx.dPdXY = length(vec2(m_viewCtx.transform.matrix.i00, m_viewCtx.transform.matrix.i10));
        m_viewCtx.zoomSpeed = 10.0f;
        m_viewCtx.sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));

        LoadScene();
    }

    __host__ void Host::GaussianSplattingModule::LoadScene()
    {
        //UnloadScene();
        
        // Component objects are interactive elements that can intercommunicate
        m_objectContainer = AssetAllocator::CreateChildAsset<Host::GenericObjectContainer>(*this, "objectcontainer");

        // The scene description contains objects used by the physically based renderer
        m_sceneContainer = AssetAllocator::CreateChildAsset<Host::SceneContainer>(*this, "scenecontainer");
        m_objectContainer->Emplace(m_sceneContainer);

        // Create some default scene objects
        Json::Node emptyDocument;
        m_viewportRenderer = AssetAllocator::CreateChildAsset<Host::ViewportRenderer>(*this, "viewportrenderer", m_objectContainer, m_clientWidth, m_clientHeight, m_renderStream);
        m_gaussianPointCloud = AssetAllocator::CreateChildAsset<Host::GaussianPointCloud>(*this, "gaussianpointcloud");
        m_objectContainer->Emplace(m_gaussianPointCloud);

        // Notify the UI that a bunch of objects has been created, then rebuild the component container
        m_sceneBuilder.Rebuild(m_sceneContainer);

        // Force a rebuild
        Rebuild(true);

        EnqueueOutboundSerialisation("OnCreateObject", kEnqueueAll);
        SignalDirty(kDirtyViewportRedraw);
    }

    __host__ void Host::GaussianSplattingModule::UnloadScene()
    {       
        m_viewportRenderer.DestroyAsset();

        //GlobalResourceRegistry::Get().Report();

        m_gaussianPointCloud.DestroyAsset();
        m_sceneContainer.DestroyAsset();

        m_renderableObjects.clear();
        m_newObject = nullptr;
        m_delegatedObject = nullptr;

        m_objectContainer.DestroyAsset();
    }

    __host__ void Host::GaussianSplattingModule::OnInboundUpdateObject(const Json::Node& node)
    {
        Assert(node.IsArray(), "Expected array");
        for (int idx = 0; idx < node.Size(); ++idx)
        {
            Json::Node itemNode = node[idx];
            Assert(itemNode.IsObject());

            std::string id;
            itemNode.GetValue("id", id, Json::kRequiredAssert | Json::kNotBlank);
            auto objectHandle = m_objectContainer->FindByID(id);

            if (!objectHandle)
            {
                Log::Warning("Error: '%s' is not a valid scene object.", id);
                continue;
            }

            objectHandle->Deserialise(itemNode, Json::kSilent);
        }
    }

    __host__ void Host::GaussianSplattingModule::EnqueueOutboundSerialisation(const std::string& eventId, const int flags, const AssetHandle<Host::GenericObject> asset)
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
            for (auto& obj : *m_objectContainer) { SerialiseImpl(nodeArray.AppendArrayObject(), obj); }
        }
        else if (flags & kEnqueueSelected)
        {
            for (auto& obj : m_selectionCtx.selectedObjects) { SerialiseImpl(nodeArray.AppendArrayObject(), obj); }
        }
        else if (flags & kEnqueueOne)
        {
            SerialiseImpl(nodeArray.AppendArrayObject(), asset);
        }

        //Log::Warning(nodeArray.Stringify(true));

        m_outboundCmdQueue->Enqueue();  // Enqueue the staged command
    }

    __host__ uint Host::GaussianSplattingModule::OnToggleRun(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        m_isRunning = !m_isRunning;
        Log::Warning(m_isRunning ? "Running" : "Paused");

        m_uiGraph.SetState("kIdleState");
        return kUIStateOkay;
    }

    __host__ uint Host::GaussianSplattingModule::OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        Log::Success("Back home!");
        return kUIStateOkay;
    }

    __host__ uint Host::GaussianSplattingModule::OnDeleteViewportObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        if (!m_selectionCtx.selectedObjects.empty())
        {
            Host::DrawableObjectContainer& drawableObjects = m_viewportRenderer->DrawableObjects();
            for (int primIdx = 0; primIdx < drawableObjects.size(); ++primIdx)
            {
                if (drawableObjects[primIdx]->IsSelected())
                {
                    // Erase the object from the container. 
                    std::lock_guard<std::mutex> lock(m_threadMutex);
                    m_deleteObjectSet.emplace(drawableObjects[primIdx]->GetAssetID());
                }
            }

            // Clear the selected object list
            DeselectAll();
        }

        return kUIStateOkay;
    }

    __host__ uint Host::GaussianSplattingModule::OnMoveViewportObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
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
        m_selectionCtx.selectedBBox.MakeInvalid();
        for (auto& obj : m_selectionCtx.selectedObjects)
        {
            Assert(obj->IsSelected());

            // Moving objects 
            if (obj->OnMove(stateID, m_viewCtx, m_selectionCtx))
            {
                m_syncObjectParamsSet.emplace(obj->GetAssetID());
                SetDirty(kDirtyViewportObjectBBox);
            }

            m_selectionCtx.selectedBBox = Union(m_selectionCtx.selectedBBox, obj->GetWorldSpaceBoundingBox());
        }

        SignalDirty(kDirtyViewportRedraw);

        if (stateID == "kMoveDrawableObjectDragging")
        {

        }

        // Enqueue the list of selected scene objects
        EnqueueOutboundSerialisation("OnUpdateObject", kEnqueueSelected);
        return kUIStateOkay;
    }

    __host__ void Host::GaussianSplattingModule::DeselectAll()
    {
        for (auto obj : m_viewportRenderer->DrawableObjects())
        {
            obj->OnSelect(false);
        }

        m_selectionCtx.selectedObjects.clear();
        m_selectionCtx.selectedBBox.MakeInvalid();

        SignalDirty(kDirtyViewportRedraw);
    }

    __host__ std::string Host::GaussianSplattingModule::DecideOnClickState(const uint& sourceStateIdx)
    {
        // Before deciding whether to lasso or move, test if the mouse has precision-clicked an object. If it has, select it.
        if (m_viewportRenderer->DrawableBIH().IsConstructed())
        {
            auto& drawableObjects = m_viewportRenderer->DrawableObjects();
            constexpr int kInvalidHit = -1;
            int hitIdx = kInvalidHit;
            uint hitResult = kDrawableObjectInvalidSelect;
            auto onContainsPrim = [&, this](const uint* primRange, const uint* primIdxs) -> bool
            {
                for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                {
                    const uint primIdx = primIdxs[idx];
                    if (primIdx >= drawableObjects.size())
                    {
                        //int size = ;
                        Log::Error("%i -> %i", primRange[0], primRange[1]);
                        for (int i = primRange[0]; i < primRange[1]; ++i)
                            Log::Error("  - %i: %i", i, primIdxs[i]);
                        Assert(false);
                    }

                    if (drawableObjects[primIdx]->GetWorldSpaceBoundingBox().Contains(m_viewCtx.mousePos))
                    {
                        hitResult = drawableObjects[primIdx]->OnMouseClick(m_keyCodes, m_viewCtx);
                        if (hitResult != kDrawableObjectInvalidSelect)
                        {
                            hitIdx = primIdx;
                            return true;
                        }
                    }
                }
                return false;
            };          
            m_viewportRenderer->DrawableBIH().TestPoint(m_viewCtx.mousePos, onContainsPrim);

            // If we've intersected something...
            if (hitIdx != kInvalidHit)
            {
                // Precision dragging instantaneously selects the object and goes into the object move state
                if (hitResult == kDrawableObjectPrecisionDrag)
                {
                    DeselectAll();

                    m_selectionCtx.selectedObjects.push_back(drawableObjects[hitIdx]);
                    m_selectionCtx.selectedObjects.back()->OnSelect(true);

                    m_selectionCtx.isLassoing = false;
                    m_selectionCtx.selectedBBox = drawableObjects[hitIdx]->GetWorldSpaceBoundingBox();

                    return "kMoveDrawableObjectBegin";
                }
                // Otherwise, start delegating mouse movements directly to the scene object until the button is lifted
                else if (hitResult == kDrawableObjectDelegatedAction)
                {
                    m_delegatedObject = drawableObjects[hitIdx];
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

    __host__ uint Host::GaussianSplattingModule::OnDelegateViewportObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
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

    __host__ void Host::GaussianSplattingModule::UpdateSelectedBBox()
    {
        m_selectionCtx.selectedBBox.MakeInvalid();
        for (auto& object : m_selectionCtx.selectedObjects)
        {
            m_selectionCtx.selectedBBox = Union(m_selectionCtx.selectedBBox, object->GetWorldSpaceBoundingBox());
        }
        SignalDirty(kDirtyViewportRedraw);
    }

    __host__ uint Host::GaussianSplattingModule::OnSelectViewportObjects(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap)
    {
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kSelectDrawableObjectDragging")
        {
            auto& drawableObjects = m_viewportRenderer->DrawableObjects();
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

            if (m_viewportRenderer->DrawableBIH().IsConstructed())
            {
                int numSelected = 0;
                auto onIntersectPrim = [&drawableObjects, this](const uint* primRange, const uint* primIdxs, const bool isInnerNode)
                {
                    // Inner nodes are tested when the bounding box envelops them completely. Hence, there's no need to do a bbox checks.
                    if (isInnerNode)
                    {
                        for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                        {
                            const uint primIdx = primIdxs[idx];
                            m_selectionCtx.selectedObjects.emplace_back(drawableObjects[primIdx]);
                            drawableObjects[primIdx]->OnSelect(true);
                        }
                    }
                    else
                    {
                        for (int idx = primRange[0]; idx < primRange[1]; ++idx)
                        {
                            const uint primIdx = primIdxs[idx];
                            const auto& bBoxWorld = drawableObjects[primIdx]->GetWorldSpaceBoundingBox();
                            const bool isCaptured = m_selectionCtx.lassoBBox.Contains(bBoxWorld);
                            if (isCaptured)
                            {
                                Log::Debug("Selected %i", primIdx);
                                m_selectionCtx.selectedObjects.emplace_back(drawableObjects[primIdx]);
                                m_selectionCtx.selectedBBox = Union(m_selectionCtx.selectedBBox, bBoxWorld);
                            }
                            drawableObjects[primIdx]->OnSelect(isCaptured);
                        }
                    }
                };
                m_viewportRenderer->DrawableBIH().TestBBox(m_selectionCtx.lassoBBox, onIntersectPrim);

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

            SignalDirty(kDirtyViewportRedraw);
            //Log::Success("Selecting!");
        }
        else if (stateID == "kSelectDrawableObjectEnd")
        {
            m_selectionCtx.isLassoing = false;
            SignalDirty(kDirtyViewportRedraw);

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
            SignalDirty(kDirtyViewportRedraw);

            //Log::Success("Finished!");
        }
        else
        {
            return kUIStateError;
        }

        return kUIStateOkay;
    }

    __host__ uint Host::GaussianSplattingModule::OnCreateViewportObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& trigger)
    {
        const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
        if (stateID == "kCreateDrawableObjectOpen")
        {
            Assert(!m_newObject); // Should be reset after object finalisation

            // Try and instantiate the objerct         
            auto newGeneric = m_componentFactory.Instantiate(trigger.HashOf(), *m_objectContainer, *this, m_objectContainer);
            m_newObject = newGeneric.DynamicCast<Host::DrawableObject>();
            m_newObject->Verify();
            m_newObjectSet.emplace(m_newObject->GetAssetID());

            SetDirty(kDirtyObjectExistence);
        }

        // Invoke the event handler of the new object
        Assert(m_newObject);
        if (m_newObject->OnCreate(stateID, m_viewCtx))
        {
            m_syncObjectParamsSet.emplace(m_newObject->GetAssetID());
            SetDirty(kDirtyViewportObjectBBox);
        }

        // Some objects will automatically finalise themselves. If this happens, we're done.
        if (m_newObject->IsFinalised())
        {
            EnqueueOutboundSerialisation("OnCreateObject", kEnqueueOne, m_newObject);
            m_newObject = nullptr;
            m_uiGraph.SetState("kIdleState");
            return kUIStateOkay;
        }

        if (stateID == "kCreateDrawableObjectClose")
        {
            FinaliseNewDrawableObject();
        }

        return kUIStateOkay;
    }

    __host__ void Host::GaussianSplattingModule::FinaliseNewDrawableObject()
    {
        Assert(m_newObject);

        // If the new object has closed but has not been finalised, delete it
        if (!m_newObject->IsFinalised())
        {
            std::lock_guard<std::mutex> lock(m_threadMutex);
            m_deleteObjectSet.emplace(m_newObject->GetAssetID());
            Log::Success("Destroying unfinalised scene object '%s'", m_newObject->GetAssetID());
        }
        else
        {
            // Serialise the new object to the outbound queue
            EnqueueOutboundSerialisation("OnCreateObject", kEnqueueOne, m_newObject);
        }

        m_newObject = nullptr;
    }

    __host__ void Host::GaussianSplattingModule::OnCommandsWaiting(CommandQueue& inbound)
    {
        m_commandManager.Flush(inbound, false);
    }

    __host__ void Host::GaussianSplattingModule::Rebuild(const bool forceRebuild)
    {
        // Lock the mutex so other threads can't interfere until we're done
        std::lock_guard<std::mutex> lock(m_threadMutex);

        bool rebuildViewport = Dirtyable::IsAnyDirty({ kDirtyObjectExistence, kDirtyViewportObjectBBox });
        bool objectsChanged = false;

        // If objects are waiting to be deleted, do so now
        if (!m_deleteObjectSet.empty())
        {
            // Release any handles that might prevent objects being successfully erased from the container
            m_viewportRenderer->ReleaseObjects();
            m_renderableObjects.clear();
            m_selectionCtx.selectedObjects.clear();

            // Iterate through the delete queue and erase the objects from the object container
            for (auto& id : m_deleteObjectSet)
            {
                m_objectContainer->Erase(id, true);
            }
            m_deleteObjectSet.clear();
            rebuildViewport = true;
            objectsChanged = true;

            // Signal to the UI that objects have been deleted
            EnqueueOutboundSerialisation("OnDeleteObject", kEnqueueAll);
        }

        // If new objects have been created, signal that the viewport should be rebuilt
        if (!m_newObjectSet.empty())
        {
            rebuildViewport = true;
            objectsChanged = true;
            m_newObjectSet.clear();
        }

        // If objects have been added or removed, do a re-bind and rebuild any data structures that depend on them
        if (objectsChanged || forceRebuild)
        {
            m_objectContainer->BindAll();
            
            m_renderableObjects.clear();
            m_objectContainer->ForEachOfType<Host::RenderableObject>([&](AssetHandle<Host::RenderableObject>& object) -> bool
                {
                    m_renderableObjects.push_back(object);
                    return true;
                });
        }

        // Some objects will signal that their parameters need to be resynced. 
        if (!m_syncObjectParamsSet.empty())
        {
            for (auto id : m_syncObjectParamsSet)
            {
                auto obj = m_objectContainer->FindByID(id);
                if (obj)
                {
                    obj->Synchronise(kSyncParams);
                }
            }
            m_syncObjectParamsSet.clear();
        }

        // Rebuild the viewport if required
        if (rebuildViewport || forceRebuild)
        {
            m_viewportRenderer->Rebuild();
        }
    }

    __host__ void Host::GaussianSplattingModule::OnRender()
    {
        // Flush any keyboard and mouse inputs that have accumulated between now and the beginning of the last frame
        FlushUIEventQueue();

        // Rebuild any objects that require it
        Rebuild(false);

        // Prepare the scene objects
        m_viewportRenderer->Prepare(m_viewCtx, m_selectionCtx, m_frameIdx);

        for (auto& object : *m_objectContainer)
        {
            AssetHandle<Host::RenderableObject> renderable = object.DynamicCast<Host::RenderableObject>();
            if (renderable)
            {
                renderable->Prepare();
                renderable->Render();
            }

            if (!object->IsClean()) 
            {
                object->Synchronise(kSyncParams);
            }
        }

        // Lock the framerate to 60fps
        //if (m_blitTimer.Get() >= 1.f / 60.f)
        {
            m_blitTimer.Reset();

            m_viewportRenderer->Render();

            // If a blit is in progress, skip the composite step entirely.
           // TODO: Make this respond intelligently to frame rate. If the CUDA renderer is running at a lower FPS than the D3D renderer then it should wait rather than
           // than skipping frames like this.
            //m_renderSemaphore.Wait(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress);
            if (!m_renderSemaphore.TryOnce(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress, false)) { return; }

            // Composite the render layers
            m_compositeImage->Clear(vec4(kZero, 1.0f));
            m_viewportRenderer->Composite(m_compositeImage);

            m_renderSemaphore.TryUntil(kRenderManagerCompInProgress, kRenderManagerCompFinished);
        }

        // Clean all the objects in the scene container ready for the next iteration
        m_objectContainer->CleanAll();
        Dirtyable::Clean();
    }

    __host__ void Host::GaussianSplattingModule::OnMouseButton(const uint code, const bool isDown)
    {
        // Is the view being changed? 
        if (isDown && (code == KEY_MBUTTON || code == KEY_RBUTTON || IsKeyDown(KEY_SHIFT)))
        {
            m_viewCtx.dragAnchor = vec2(m_mouse.pos);
            m_viewCtx.rotAxis = normalize(m_viewCtx.dragAnchor - vec2(m_clientWidth, m_clientHeight) * 0.5f);
            m_viewCtx.transAnchor = m_viewCtx.transform.trans;
            m_viewCtx.scaleAnchor = m_viewCtx.transform.scale;
            m_viewCtx.rotAnchor = m_viewCtx.transform.rotate;
        }
    }

    __host__ void Host::GaussianSplattingModule::OnMouseMove()
    {
        OnViewChange();

        m_viewCtx.mousePos = m_viewCtx.transform.matrix * vec2(m_mouse.pos);
    }

    __host__ void Host::GaussianSplattingModule::OnViewChange()
    {
        auto& transform = m_viewCtx.transform;
        bool isUpdated = true;

        // Zooming?
        if (IsMouseButtonDown(KEY_RBUTTON))
        {
            float logScaleAnchor = std::log2(std::max(1e-10f, m_viewCtx.scaleAnchor));
            logScaleAnchor += m_viewCtx.zoomSpeed * float(m_mouse.pos.y - m_viewCtx.dragAnchor.y) / m_clientHeight;
            transform.scale = std::pow(2.0, logScaleAnchor);

            //Log::Write("Scale: %f", transform.scale);
        }
        // Rotating?
        else if (IsKeyDown(KEY_SHIFT) && IsMouseButtonDown(KEY_LBUTTON))
        {
            const vec2 delta = normalize(vec2(m_mouse.pos) - vec2(m_clientWidth, m_clientHeight) * 0.5f);
            const float theta = std::acos(dot(delta, m_viewCtx.rotAxis)) * (float(dot(delta, vec2(m_viewCtx.rotAxis.y, -m_viewCtx.rotAxis.x)) < 0.0f) * 2.0 - 1.0f);
            transform.rotate = m_viewCtx.rotAnchor + theta;

            if (std::abs(std::fmod(transform.rotate, kHalfPi)) < 0.05f) { transform.rotate = std::round(transform.rotate / kHalfPi) * kHalfPi; }

            //Log::Write("Theta: %f", transform.rotate);
        }
        // Translating
        else if (IsMouseButtonDown(KEY_MBUTTON))
        {
            // Update the transformation
            const mat3 newMat = ConstructViewMatrix(m_viewCtx.transAnchor, transform.rotate, transform.scale) * m_clientToNormMatrix;
            const vec2 dragDelta = (newMat * vec2(m_viewCtx.dragAnchor)) - (newMat * vec2(m_mouse.pos));
            transform.trans = m_viewCtx.transAnchor + dragDelta;

            //Log::Write("Trans: %s", m_viewCtx.trans.format());
        }
        else
        {
            isUpdated = false;
        }

        // Update the parameters in the overlay renderer
        if (isUpdated)
        {
            transform.matrix = ConstructViewMatrix(transform.trans, transform.rotate, transform.scale) * m_clientToNormMatrix;
            m_viewCtx.Prepare();

            // Mark the scene as dirty
            SignalDirty(kDirtyViewportRedraw);
        }
    }

    __host__ void Host::GaussianSplattingModule::OnResizeClient()
    {
    }

    __host__ void Host::GaussianSplattingModule::OnFocusChange(const bool isSet)
    {
        // Finalise any objects that are in the process of being created
        /*if (m_newObject)
        {
            FinaliseNewDrawableObject();
        }*/
    }

    __host__ bool Host::GaussianSplattingModule::Serialise(Json::Document& json, const int flags)
    {
        return true;
    }
}