#include "RendererManager.h"

#include "gi2d/GI2D.h"

RendererManager::RendererManager()
{
    AddInstantiator<GI2D>("2dgi");

    
}

void RendererManager::Destroy()
{
    UnloadRenderer();
    
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
    m_activeRenderer->Initialise();

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