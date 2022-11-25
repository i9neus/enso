#pragma once

#include "../ModuleInterface.h"
#include "UICtx.cuh"
#include "core/Vector.cuh"
#include "FwdDecl.cuh"
#include "core/GenericObjectFactory.cuh"

namespace Enso
{
    class Asset;
    class GenericObjectContainer;
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
        GI2DRenderer();
        virtual ~GI2DRenderer();

        virtual void OnInitialise() override final;
        virtual void OnMouseMove() override final;
        virtual void OnMouseButton(const uint code, const bool isDown) override final;
        virtual void OnMouseWheel() override final;
        virtual void OnKey(const uint code, const bool isSysKey, const bool isDown) override final;
        virtual void OnResizeClient() override final;

        //virtual void OnResizeClient() override final;
        virtual std::string GetRendererName() const { return "2D GI Sandbox"; };

        static std::shared_ptr<ModuleInterface> Instantiate();

    private:
        virtual void            OnDestroy() override final;
        //virtual void            OnPreRender() override final;
        virtual void            OnRender() override final;
        //virtual void          OnPostRender() override final;
        void                    OnViewChange();

        void                    Rebuild();

        uint                    OnMoveSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        uint                    OnCreateSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        uint                    OnSelectSceneObjects(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        uint                    OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        uint                    OnDeleteSceneObject(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);
        uint                    OnToggleRun(const uint& sourceStateIdx, const uint& targetStateIdx, const VirtualKeyMap& keyMap);

        std::string             DecideOnClickState(const uint& sourceStateIdx);
        void                    DeselectAll();

    private:
        enum JobIDs : uint { kJobDraw };

        AssetHandle<Host::OverlayLayer>       m_overlayRenderer;
        AssetHandle<Host::PathTracerLayer>    m_pathTracerLayer;
        AssetHandle<Host::VoxelProxyGridLayer> m_voxelProxyGridLayer;
        //AssetHandle<Host::IsosurfaceExplorer> m_isosurfaceExplorer;

        AssetHandle<Host::SceneDescription>   m_scene;
        //std::vector<BBox2f>                   m_tracableBBoxes;

        std::unique_ptr<ViewTransform2D>      m_viewTransform;

        AssetHandle<GenericObjectContainer> m_renderObjects;

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

        bool                                        m_isRunning;
        HighResolutionTimer                         m_renderTimer;
    };
}