#pragma once

#include "../ModuleBase.cuh"
#include "core/2d/Ctx.cuh"
#include "core/Vector.cuh"
#include "FwdDecl.cuh"
#include "core/GenericObjectFactory.cuh"
#include "io/CommandManager.h"

namespace Enso
{
    class Asset;
    namespace Host
    {
        class RenderObject;
        //template<typename T> class Vector;

        struct CudaObjects;        

        class GaussianSplattingModule : public ModuleBase
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

            __host__ virtual void OnDirty(const DirtinessEvent& flag, WeakAssetHandle<Host::Asset>& caller) override final;

            //__host__ virtual void OnResizeClient() override final;
            __host__ virtual std::string GetRendererName() const { return "Gaussian Splatting Sandbox"; };

            __host__ static AssetHandle<ModuleBase> Instantiate(std::shared_ptr<CommandQueue> outQueue);

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

            __host__ void                    Rebuild(const bool forceRebuild);
            __host__ void                    LoadScene();
            __host__ void                    UnloadScene();

            __host__ uint                    OnMoveViewportObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnCreateViewportObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnSelectViewportObjects(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnDelegateViewportObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnDeleteViewportObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnToggleRun(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ void                    OnInboundUpdateObject(const Json::Node& node);

            __host__ std::string             DecideOnClickState(const uint& sourceStateIdx);
            __host__ void                    DeselectAll();

            __host__ void                    EnqueueOutboundSerialisation(const std::string& eventId, const int flags, const AssetHandle<Host::GenericObject> asset = nullptr);
            __host__ void                    FinaliseNewDrawableObject();
            __host__ void                    UpdateSelectedBBox();


        private:
            enum JobIDs : uint { kJobDraw };
            
            // Primary repository for all objects created by the module 
            AssetHandle<Host::GenericObjectContainer>           m_objectContainer;

            // Renderable objects designed to be cycled rapidly by the inner loop
            std::vector<AssetHandle<Host::RenderableObject>>    m_renderableObjects;

            // Viewport renderer
            AssetHandle<Host::ViewportRenderer>     m_viewportRenderer;

            AssetHandle<Host::SceneContainer>       m_sceneContainer;
            AssetHandle<Host::SceneBuilder>         m_sceneBuilder;

            UIGridCtx                               m_gridCtx;
            UIViewCtx                               m_viewCtx;
            UISelectionCtx                          m_selectionCtx;

            Host::GenericObjectFactory<const Host::Asset&, const AssetHandle<const Host::GenericObjectContainer>&> m_componentFactory;

            AssetHandle<Host::DrawableObject>           m_newObject;

            std::mutex                                  m_threadMutex;
            std::unordered_set<std::string>             m_deleteObjectQueue;
            std::unordered_set<std::string>             m_rebuildObjectQueue;
            std::unordered_set<std::string>             m_newObjectQueue;

            AssetHandle<Host::DrawableObject>           m_delegatedObject;
            CommandManager                              m_commandManager;

            bool                                        m_isRunning;
            HighResolutionTimer                         m_blitTimer;
            
        };
    }
}