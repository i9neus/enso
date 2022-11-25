#pragma once

#include "ModuleInterface.h"

namespace Enso
{
    struct RendererComponentInfo
    {
        std::string id;
        std::string name;
    };

    class ModuleManager : public CudaObjectManager
    {
    public:
        ModuleManager();

        void                                    Initialise(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight);
        void                                    Destroy();
        std::vector<RendererComponentInfo>      GetRendererList() const;
        ModuleInterface& GetRenderer() { Assert(m_activeRenderer); return *m_activeRenderer; }
        std::shared_ptr<ModuleInterface>      GetRendererPtr() { return m_activeRenderer; }
        void                                    LoadRenderer(const std::string& id);
        void                                    UnloadRenderer();
        void						            UpdateD3DOutputTexture(UINT64& currentFenceValue);

        unsigned int diag[2];

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
        std::shared_ptr<ModuleInterface>                                                      m_activeRenderer;
        std::unordered_map<std::string, std::function<std::shared_ptr<ModuleInterface>()>>    m_instantiators;

    private:
        AssetHandle<Host::ImageRGBA>		m_compositeImage;
    };
}