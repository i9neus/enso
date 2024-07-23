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

        enum GI2DEditMode : int
        {
            kGI2DIdle,
            kGI2DAddPath
        };

        enum GI2DAddPath : int
        {
            kGI2DAddPathHovering,
            kGI2DAddPathPositioning,
            kGI2DAddPathFinished
        };

        class GI2DModule : public Host::Dirtyable,
            public ModuleBase
        {
        public:
            enum EnqueueFlags : int { kEnqueueOne = 1, kEnqueueSelected = 2, kEnqueueAll = 4, kEnqueueIdOnly = 8 };

            __host__ GI2DModule(const InitCtx& initCtx, std::shared_ptr<CommandQueue> outQueue);
            __host__ virtual ~GI2DModule() noexcept;

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

            __host__ void                    LoadScene();

            __host__ uint                    OnMoveSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnCreateSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnSelectSceneObjects(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnDelegateSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnDeleteSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ uint                    OnToggleRun(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
            __host__ void                    OnInboundUpdateObject(const Json::Node& node);

            __host__ std::string             DecideOnClickState(const uint& sourceStateIdx);
            __host__ void                    DeselectAll();

            __host__ void                    EnqueueOutboundSerialisation(const std::string& eventId, const int flags, const AssetHandle<Host::GenericObject> asset = nullptr);
            __host__ void                    FinaliseNewSceneObject();
            __host__ void                    UpdateSelectedBBox();


        private:
            enum JobIDs : uint { kJobDraw };

            AssetHandle<Host::OverlayLayer>       m_overlayRenderer;
            AssetHandle<Host::VoxelProxyGrid>     m_voxelProxyGrid;

            AssetHandle<Host::SceneContainer>     m_sceneContainer;
            AssetHandle<Host::SceneBuilder>       m_sceneBuilder;

            std::unique_ptr<ViewTransform2D>      m_viewTransform;

            UIGridCtx                             m_gridCtx;
            UIViewCtx                             m_viewCtx;
            UISelectionCtx                        m_selectionCtx;

            Host::GenericObjectFactory<const Host::Asset&, const AssetHandle<const Host::SceneContainer>&> m_sceneObjectFactory;

            struct
            {
                AssetHandle<Host::SceneObject>   newObject;
            }
            m_onCreate;

            AssetHandle<Host::SceneObject>              m_delegatedObject;
            CommandManager                              m_commandManager;

            bool                                        m_isRunning;
            HighResolutionTimer                         m_renderTimer;
        };
    }
}