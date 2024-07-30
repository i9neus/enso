#include "ModuleManager.cuh"
#include "core/HighResolutionTimer.h"
#include "io/CommandQueue.h"

#include "gi2d/GI2DModule.cuh"
#include "gaussiansplatting/GaussianSplattingModule.cuh"

namespace Enso
{
    ModuleManager::ModuleManager() : 
        m_outboundCmdQueue(std::make_shared<CommandQueue>()),
        m_parentWnd(0)
    {
        AddInstantiator<Host::GI2DModule>("2dgi");
        AddInstantiator<Host::GaussianSplattingModule>("gaussiansplatting");
    }

    void ModuleManager::Initialise(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight, HWND hWnd)
    {
        m_parentWnd = hWnd;
        InitialiseCuda(dx12DeviceLUID, clientWidth, clientHeight);

        // Create some Cuda objects
        m_compositeImage = AssetAllocator::CreateAsset<Host::ImageRGBA>("id_compositeImage", clientWidth, clientHeight, m_renderStream);
    }

    void ModuleManager::Destroy()
    {
        UnloadRenderer();

        // Destroy assets
        m_compositeImage.DestroyAsset();

        DestroyCuda();
    }

    std::vector<RendererComponentInfo> ModuleManager::GetRendererList() const
    {
        std::vector<RendererComponentInfo> rendererList;
        for (const auto& renderer : m_instantiators)
        {
            rendererList.push_back(RendererComponentInfo{ renderer.first, renderer.first });
        }

        return rendererList;
    }

    void ModuleManager::UpdateD3DOutputTexture(UINT64& currentFenceValue)
    {
        if (m_activeRenderer->GetRenderSemaphore().Try(kRenderManagerCompFinished, kRenderManagerD3DBlitInProgress, false))
        {
            HighResolutionTimer timer;
            CudaObjectManager::UpdateD3DOutputTexture(currentFenceValue, m_compositeImage, true);
            m_activeRenderer->GetRenderSemaphore().Try(kRenderManagerD3DBlitInProgress, kRenderManagerD3DBlitFinished, true);
            //Log::System("Blitted");       
        }
        else
        {
            HighResolutionTimer timer;
            CudaObjectManager::UpdateD3DOutputTexture(currentFenceValue, m_compositeImage, false);
            //Log::Warning("Not blitted");
        }
    }

    /*std::shared_ptr<ModuleInterface> ModuleManager::GetRenderer()
    {
        Assert(m_activeRenderer);
        return m_activeRenderer;
    }*/

    void ModuleManager::LoadRenderer(const std::string& id)
    {
        UnloadRenderer();

        auto it = m_instantiators.find(id);
        AssertMsgFmt(it != m_instantiators.end(), "Requested renderer '%s' is invalid.", id.c_str());

        Log::Indent indent("Loading renderer...");

        // Instantiate and set up the renderer
        m_activeRenderer = (it->second)(m_outboundCmdQueue);
        m_activeRenderer->SetCudaObjects(m_compositeImage, m_renderStream);
        m_activeRenderer->Initialise(m_clientWidth, m_clientHeight, m_parentWnd);

        Log::Success("Successfully loaded '%s'!", id);
    }

    void ModuleManager::UnloadRenderer()
    {
        if (!m_activeRenderer) { return; }

        AssertMsgFmt(m_activeRenderer.use_count() == 1, "Renderer object has %i active references.", m_activeRenderer.use_count());

        Log::Indent indent("Unloading renderer...");

        m_activeRenderer->Stop();

        m_activeRenderer->Destroy();
        
        m_activeRenderer.reset();

        Log::Success("Successfully unloaded renderer!");
    }

    bool ModuleManager::Serialise(Json::Document& json, const int flags)
    {
        return m_activeRenderer->Serialise(json, flags);
    }

    void ModuleManager::SetInboundCommandQueue(std::shared_ptr<CommandQueue> inbound)
    { 
        m_inboundCmdQueue = inbound; 
        m_activeRenderer->SetInboundCommandQueue(inbound);
    }

}