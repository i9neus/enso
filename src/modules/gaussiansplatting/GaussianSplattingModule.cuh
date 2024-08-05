#pragma once

#include "../ModuleBase.cuh"
#include "io/CommandManager.h"
#include "core/GenericObjectFactory.cuh"
#include "FwdDecl.cuh"

namespace Enso
{
    namespace Host
    {
        class GaussianSplattingModule : public Host::Dirtyable, public ModuleBase
        {
        public:
            enum EnqueueFlags : int { kEnqueueOne = 1, kEnqueueSelected = 2, kEnqueueAll = 4, kEnqueueIdOnly = 8 };

            __host__ GaussianSplattingModule(const InitCtx& initCtx, std::shared_ptr<CommandQueue> outQueue);
            __host__ virtual ~GaussianSplattingModule() noexcept;

            __host__ virtual void OnInitialise() override final;
            __host__ virtual void OnMouseMove() override final;
            __host__ virtual void OnMouseButton(const uint code, const bool isDown) override final;
            __host__ virtual void OnResizeClient() override final;
            __host__ virtual void OnFocusChange(const bool isSet) override final;

            __host__ virtual void OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller) override final;

            //__host__ virtual void OnResizeClient() override final;
            __host__ virtual std::string GetRendererName() const { return "2D GI Sandbox"; };

            __host__ static std::shared_ptr<ModuleBase> Instantiate(std::shared_ptr<CommandQueue> outQueue);

            __host__ virtual bool Serialise(Json::Document& json, const int flags) override final;

        private:
            //virtual void            OnPreRender() override final;
            __host__ virtual void            OnRender() override final;
            //virtual void          OnPostRender() override final;
            __host__ void                    OnViewChange();
            __host__ void                    OnCommandsWaiting(CommandQueue& inbound) override final;

            __host__ void                    RegisterInstantiators();
            __host__ void                    DeclareStateTransitionGraph();
            __host__ void                    DeclareListeners();
            __host__ void                    UpdatePerfStats();

            __host__ void                    LoadScene();
        
            __host__ uint                    OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnToggleRun(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ void                    OnInboundUpdateObject(const Json::Node& node);

            __host__ void                    EnqueueOutboundSerialisation(const std::string& eventId, const int flags, const AssetHandle<Host::GenericObject> asset = nullptr);

        private:
            CommandManager                              m_commandManager;

            AssetHandle<Host::PathTracer>               m_pathTracer;

            bool                                        m_isRunning;
            HighResolutionTimer                         m_renderTimer;
            HighResolutionTimer                         m_blitTimer;
            std::array<float, 60>                       m_timeRingBuffer;
            int                                         m_timeRingIdx;
        };
    }
}