#include "GaussianSplattingModule.cuh"

#include "core/math/Math.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/GenericObjectContainer.cuh"
#include "core/Vector.cuh"
#include "core/Tuple.cuh"

#include "io/SerialisableObjectSchema.h"

#include "pathtracer/PathTracer.cuh"

//#include "kernels/gi2d/ObjectDebugger.cuh"

namespace Enso
{

    __host__ Host::GaussianSplattingModule::GaussianSplattingModule(const InitCtx& initCtx, std::shared_ptr<CommandQueue> outQueue) :
        Dirtyable(initCtx),
        ModuleBase(outQueue),
        m_isRunning(true)
    {
        // Load the object schema
        SerialisableObjectSchemaContainer::Load("schema.json");

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

        // Zero the ring buffer
        m_timeRingIdx = 0;
        for (auto& t : m_timeRingBuffer) { t = 0.f; }
    }

    __host__ Host::GaussianSplattingModule::~GaussianSplattingModule() noexcept
    {
        m_pathTracer.DestroyAsset();
    }

    __host__ void Host::GaussianSplattingModule::RegisterInstantiators()
    {
  
    }

    __host__ void Host::GaussianSplattingModule::DeclareStateTransitionGraph()
    {
        m_uiGraph.DeclareState("kIdleState", this, &Host::GaussianSplattingModule::OnIdleState);

        m_uiGraph.Finalise();
    }

    __host__ void Host::GaussianSplattingModule::DeclareListeners()
    {

    }

    __host__ void Host::GaussianSplattingModule::OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller)
    {
        SetDirty(flag);
    }

    std::shared_ptr<ModuleBase> Host::GaussianSplattingModule::Instantiate(std::shared_ptr<CommandQueue> outQueue)
    {
        AssetHandle<ModuleBase> newAsset = AssetAllocator::CreateAsset<Host::GaussianSplattingModule>("gaussiansplatting", outQueue);
        return std::shared_ptr<ModuleBase>(newAsset);
    }

    __host__ void Host::GaussianSplattingModule::OnInitialise()
    {
        LoadScene();

        m_blitTimer.Reset();
        m_renderTimer.Reset();
    }

    __host__ void Host::GaussianSplattingModule::LoadScene()
    {
        m_pathTracer = AssetAllocator::CreateChildAsset<Host::PathTracer>(*this, "pathTracer", 1200, 675, m_renderStream);

        EnqueueOutboundSerialisation("OnCreateObject", kEnqueueAll);
    }

    __host__ void Host::GaussianSplattingModule::OnInboundUpdateObject(const Json::Node& node)
    {
        Assert(node.IsArray(), "Expected array");
        for(int idx = 0; idx < node.Size(); ++idx)
        {
            Json::Node itemNode = node[idx];
            Assert(itemNode.IsObject());

            std::string id;
            itemNode.GetValue("id", id, Json::kRequiredAssert | Json::kNotBlank);

            /*auto objectHandle = m_sceneContainer->GenericObjects().FindByID(id);

            if (!objectHandle)
            {
                Log::Warning("Error: '%s' is not a valid scene object.", id);
                continue;
            }

            objectHandle->Deserialise(itemNode, Json::kSilent);*/
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
            //for (auto& obj : m_sceneContainer->GenericObjects()) { SerialiseImpl(nodeArray.AppendArrayObject(), obj); }
        }
        else if (flags & kEnqueueSelected)
        {
            //for (auto& obj : m_selectionCtx.selectedObjects) { SerialiseImpl(nodeArray.AppendArrayObject(), obj); }
        }
        else if (flags & kEnqueueOne)
        {
            SerialiseImpl(nodeArray.AppendArrayObject(), asset);
        }

        Log::Warning(nodeArray.Stringify(true));

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

    __host__ void Host::GaussianSplattingModule::OnCommandsWaiting(CommandQueue& inbound)
    {
        m_commandManager.Flush(inbound, false);
    }

    __host__ void Host::GaussianSplattingModule::OnRender()
    {        
        // Flush any keyboard and mouse inputs that have accumulated between now and the beginning of the last frame
        FlushUIEventQueue();

        m_renderTimer.Reset();

        m_pathTracer->Prepare();
        m_pathTracer->Render();

        //UpdatePerfStats();

        Clean();

        // Lock the framerate to 60fps
        if (m_blitTimer.Get() >= 1.f / 60.f)
        {
            m_blitTimer.Reset();

            // If a blit is in progress, skip the composite step entirely.
            // TODO: Make this respond intelligently to frame rate. If the CUDA renderer is running at a lower FPS than the D3D renderer then it should wait rather than
            // than skipping frames like this.
            m_renderSemaphore.Wait(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress);
            //if (!m_renderSemaphore.Try(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress, false)) { return; }

            // Composite the render layers
            m_compositeImage->Clear(vec4(kZero, 1.0f));
            m_pathTracer->Composite(m_compositeImage);

            m_renderSemaphore.Wait(kRenderManagerCompInProgress, kRenderManagerCompFinished);
            //m_renderSemaphore.Try(kRenderManagerCompInProgress, kRenderManagerCompFinished, true);
        }
    }

    __host__ void Host::GaussianSplattingModule::UpdatePerfStats()
    {
        // Compute the exponential moving average of the frame time
        m_timeRingBuffer[m_timeRingIdx] = m_renderTimer.Get();
        m_timeRingIdx = (m_timeRingIdx + 1) % m_timeRingBuffer.size();
        float meanTime = 0., sumWeights = 0.;
        constexpr float kEMASigma = 1.5f;
        for (int i = 0, j = m_timeRingIdx - 1; i < m_timeRingBuffer.size(); ++i, --j)
        {
            const float weight = std::exp(-sqr(kEMASigma * float(i) / float(m_timeRingBuffer.size())));
            meanTime += weight * m_timeRingBuffer[(j >= 0) ? j : (m_timeRingBuffer.size() + j)];
            sumWeights += weight;
        }
        meanTime /= sumWeights;

        std::string windowTitle = tfm::format("%i fps (%.2 ms/frame)", int(1. / meanTime), meanTime * 1e3f);
        SetWindowTextA(m_parentWnd, windowTitle.c_str());
    }

    __host__ void Host::GaussianSplattingModule::OnMouseButton(const uint code, const bool isDown)
    {
       
    }

    __host__ void Host::GaussianSplattingModule::OnMouseMove()
    {
       
    }

    __host__ void Host::GaussianSplattingModule::OnViewChange()
    {
        
    }

    __host__ void Host::GaussianSplattingModule::OnResizeClient()
    {
    } 

    __host__ void Host::GaussianSplattingModule::OnFocusChange(const bool isSet)
    {
        // Finalise any objects that are in the process of being created
        /*if (m_onCreate.newObject)
        {
            FinaliseNewSceneObject();
        }*/
    }

    __host__ bool Host::GaussianSplattingModule::Serialise(Json::Document& json, const int flags)
    {
        return true;
    }
}