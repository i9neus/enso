#include "RendererManager.h"

#include "GI2D.h"

RendererManager::RendererManager()
{
    AddInstantiator<GI2D>("gi2d");

    
}

void RendererManager::Destroy()
{
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
    AssertMsgFmt(it != m_instantiators.end(), "Requested renderer '%s' is invalid.", id);

    // Instantiate 
    m_activeRenderer = (it->second)();

    m_activeRenderer->SetCudaObjects(m_compositeImage, m_renderStream);
}

void RendererManager::UnloadRenderer()
{
    if (!m_activeRenderer) { return; }

    AssertMsgFmt(m_activeRenderer.use_count() == 1,  "Active renderer object has %i active references.", m_activeRenderer.use_count());

    m_activeRenderer->Destroy();
    m_activeRenderer.reset();
}