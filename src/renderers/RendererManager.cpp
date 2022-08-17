#include "RendererManager.h"

#include "gi2d/GI2D.cuh"

RendererManager::RendererManager()
{
    AddInstantiator<GI2DRenderer>("2dgi");
}

void RendererManager::Initialise(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight)
{
    InitialiseCuda(dx12DeviceLUID, clientWidth, clientHeight);

    // Create some Cuda objects
    m_compositeImage = Cuda::CreateAsset<Cuda::Host::ImageRGBA>("id_compositeImage", clientWidth, clientHeight, m_renderStream);
}

void RendererManager::Destroy()
{
    UnloadRenderer();

    // Destroy assets
    m_compositeImage.DestroyAsset();
    
    DestroyCuda();
}

std::vector<RendererComponentInfo> RendererManager::GetRendererList() const
{    
    std::vector<RendererComponentInfo> rendererList;
    for (const auto& renderer : m_instantiators)
    {
        rendererList.push_back(RendererComponentInfo{ renderer.first, renderer.first });
    } 
     
    return rendererList;
}

void RendererManager::UpdateD3DOutputTexture(UINT64& currentFenceValue)
{
    diag[0] = (unsigned int)(m_activeRenderer->GetRenderSemaphore());
    
    if (m_activeRenderer->GetRenderSemaphore().Try(kRenderManagerCompFinished, kRenderManagerD3DBlitInProgress, false))
    {
        Timer timer;
        CudaObjectManager::UpdateD3DOutputTexture(currentFenceValue, m_compositeImage, true);
        m_activeRenderer->GetRenderSemaphore().Try(kRenderManagerD3DBlitInProgress, kRenderManagerD3DBlitFinished, true);        
        //Log::System("Blitted");       
    }
    else
    {       
        Timer timer;
        CudaObjectManager::UpdateD3DOutputTexture(currentFenceValue, m_compositeImage, false);
        //Log::Warning("Not blitted");
    }

    diag[1] = (unsigned int)(m_activeRenderer->GetRenderSemaphore());
}

/*std::shared_ptr<RendererInterface> RendererManager::GetRenderer()
{
    Assert(m_activeRenderer);    
    return m_activeRenderer;
}*/

void RendererManager::LoadRenderer(const std::string& id)
{
    UnloadRenderer();

    auto it = m_instantiators.find(id);
    AssertMsgFmt(it != m_instantiators.end(), "Requested renderer '%s' is invalid.", id.c_str());

    Log::Indent indent("Loading renderer...");

    // Instantiate and set up the renderer
    m_activeRenderer = (it->second)();
    m_activeRenderer->SetCudaObjects(m_compositeImage, m_renderStream);
    m_activeRenderer->Initialise(m_clientWidth, m_clientHeight);

    Log::Success("Successfully loaded '%s'!", id);
}

void RendererManager::UnloadRenderer()
{
    if (!m_activeRenderer) { return; }

    AssertMsgFmt(m_activeRenderer.use_count() == 1,  "Renderer object has %i active references.", m_activeRenderer.use_count());

    Log::Indent indent("Unloading renderer...");

    m_activeRenderer->Stop();

    m_activeRenderer->Destroy();
    m_activeRenderer.reset();

    Log::Success("Successfully unloaded renderer!");
}