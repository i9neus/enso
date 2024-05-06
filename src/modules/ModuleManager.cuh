#pragma once

#include "ModuleBase.cuh"

namespace Enso
{
    struct RendererComponentInfo
    {
        std::string id;
        std::string name;
    };

    enum ModuleManagerSerialiseFlags : int
    {
        kModuleManagerSerialiseAll,
        kModuleManagerSerialiseDirty
    };

    class ModuleManager : public CudaObjectManager
    {
    public:
        ModuleManager();

        void                                    Initialise(const LUID& dx12DeviceLUID, const UINT clientWidth, const UINT clientHeight);
        void                                    Destroy();
        std::vector<RendererComponentInfo>      GetRendererList() const;
        ModuleBase& GetRenderer() { Assert(m_activeRenderer); return *m_activeRenderer; }
        std::shared_ptr<ModuleBase>        GetRendererPtr() { return m_activeRenderer; }
        void                                    LoadRenderer(const std::string& id);
        void                                    UnloadRenderer();
        void						            UpdateD3DOutputTexture(UINT64& currentFenceValue);

        inline bool                             Serialise(Json::Document& json, const int flags);
        std::shared_ptr<CommandQueue>           GetOutboundCommandQueue() { return m_outboundCmdQueue; }
        void                                    SetInboundCommandQueue(std::shared_ptr<CommandQueue> inbound);

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
        using Instantiator = std::function<std::shared_ptr<ModuleBase>(std::shared_ptr<CommandQueue>)>;
        std::shared_ptr<ModuleBase>  m_activeRenderer;
        std::unordered_map<std::string, Instantiator> m_instantiators;

    private:
        AssetHandle<Host::ImageRGBA>		m_compositeImage;

        std::shared_ptr<CommandQueue>       m_inboundCmdQueue;
        std::shared_ptr<CommandQueue>       m_outboundCmdQueue;
    };
}