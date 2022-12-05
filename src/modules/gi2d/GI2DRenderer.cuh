#pragma once

#include "../ModuleInterface.h"
#include "UICtx.cuh"
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
    }

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

    class GI2DRenderer : public ModuleInterface
    {
    public:
        enum EnqueueFlags : int { kEnqueueOne = 1, kEnqueueSelected = 2, kEnqueueAll = 4, kEnqueueIdOnly = 8 };

        __host__ GI2DRenderer(std::shared_ptr<CommandQueue> outQueue);
        __host__ virtual ~GI2DRenderer();

        __host__ virtual void OnInitialise() override final;
        __host__ virtual void OnMouseMove() override final;
        __host__ virtual void OnMouseButton(const uint code, const bool isDown) override final;
        __host__ virtual void OnMouseWheel() override final;
        __host__ virtual void OnKey(const uint code, const bool isSysKey, const bool isDown) override final;
        __host__ virtual void OnResizeClient() override final;
        __host__ virtual void OnFocusChange(const bool isSet) override final;


        //__host__ virtual void OnResizeClient() override final;
        __host__ virtual std::string GetRendererName() const { return "2D GI Sandbox"; };

        __host__ static std::shared_ptr<ModuleInterface> Instantiate(std::shared_ptr<CommandQueue> outQueue);

        __host__ virtual bool Serialise(Json::Document& json, const int flags) override final;

    private:
        __host__ virtual void            OnDestroy() override final;
        //virtual void            OnPreRender() override final;
        __host__ virtual void            OnRender() override final;
        //virtual void          OnPostRender() override final;
        __host__ void                    OnViewChange();
        __host__ void                    OnCommandsWaiting(CommandQueue& inbound) override final;

        __host__ void                    Rebuild();

        __host__ uint                    OnMoveSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        __host__ uint                    OnCreateSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        __host__ uint                    OnSelectSceneObjects(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        __host__ uint                    OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        __host__ uint                    OnDeleteSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        __host__ uint                    OnToggleRun(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);

        __host__ std::string             DecideOnClickState(const uint& sourceStateIdx);
        __host__ void                    DeselectAll();

        __host__ void                    EnqueueObjects(const std::string& eventId, const int flags, const AssetHandle<Host::SceneObject> asset = nullptr);
        __host__ void                    FinaliseNewSceneObject();


    private:
        enum JobIDs : uint { kJobDraw };

        AssetHandle<Host::OverlayLayer>       m_overlayRenderer;
        AssetHandle<Host::PathTracerLayer>    m_pathTracerLayer;
        AssetHandle<Host::VoxelProxyGridLayer> m_voxelProxyGridLayer;
        //AssetHandle<Host::IsosurfaceExplorer> m_isosurfaceExplorer;

        AssetHandle<Host::SceneDescription>   m_scene;
        //std::vector<BBox2f>                   m_tracableBBoxes;

        std::unique_ptr<ViewTransform2D>      m_viewTransform;

        AssetHandle<GenericObjectContainer>   m_sceneObjects;

        UIGridCtx                             m_gridCtx;
        UIViewCtx                             m_viewCtx;
        UISelectionCtx                        m_selectionCtx;

        GenericObjectFactory                  m_sceneObjectFactory;

        struct
        {
            AssetHandle<Host::SceneObject>   newObject;
        }
        m_onCreate;

        struct
        {
            vec2                              dragAnchor;
        }
        m_onMove;

        std::vector<AssetHandle<Host::Tracable>>    m_selectedTracables;
        CommandManager                              m_commandManager;

        bool                                        m_isRunning;
        HighResolutionTimer                         m_renderTimer;
    };
}