#pragma once

#include "RendererInterface.h"

struct RendererComponentInfo
{
    std::string id;
    std::string name;
};

class RendererManager : public CudaObjectManager
{
public:
    RendererManager();

    void                                    Destroy();

    std::vector<RendererComponentInfo>      GetRendererList() const;

    std::shared_ptr<RendererInterface>      GetRenderer() { Assert(m_activeRenderer); return m_activeRenderer; }
    void                                    LoadRenderer(const std::string& id);
    void                                    UnloadRenderer();

private:
    template<typename HostClass>
    void AddInstantiator(const std::string& id)
    {
        auto it = m_instantiators.find(id);
        AssertMsgFmt(it == m_instantiators.end(),
            "Internal error: a renderer instantiator with ID '%s' already exists.\n", id);

        m_instantiators[id] = HostClass::Instantiate;
    }

public:
    std::shared_ptr<RendererInterface>                                                      m_activeRenderer;
    std::unordered_map<std::string, std::function<std::shared_ptr<RendererInterface>()>>    m_instantiators;
};