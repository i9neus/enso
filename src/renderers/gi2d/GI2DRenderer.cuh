#pragma once

#include "../RendererInterface.h"
#include "generic/Job.h"
#include "kernels/gi2d/UICtx.cuh"
#include "kernels/CudaVector.cuh"
#include "kernels/gi2d/SceneDescription.cuh"

namespace GI2D
{
    struct OverlayLayerParams;  
    struct PathTracerLayerParams; 
    struct ViewTransform2D;
    //class LineSegment; 

    namespace Device
    {
        class Tracable;
    }

    namespace Host
    {
        class BIH2DAsset;
        class OverlayLayer;   
        class PathTracerLayer; 
        class VoxelProxyGrid;
        //class IsosurfaceExplorer;
        class TracableInterface;
        class SceneObjectInterface;
    }
}

namespace Cuda
{
    class Asset;
    class RenderObjectContainer;
    namespace Host
    {
        class RenderObject;
        //template<typename T> class Vector;
    }
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

class GI2DRenderer : public RendererInterface
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

    static std::shared_ptr<RendererInterface> Instantiate();

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

    std::string             DecideOnClickState(const uint& sourceStateIdx);
    void                    DeselectAll();

    template<typename HostClass>
    __host__ void AddInstantiator(const uint keyCode)
    {
        //const auto id = HostClass::GetAssetTypeString();
        auto it = m_sceneObjectInstantiators.find(keyCode);
        Assert(it == m_sceneObjectInstantiators.end());
        //AssertMsgFmt(it == m_sceneObjectInstantiators.end(), "Internal error: a render object instantiator with ID '%s' already exists.\n", id);

        m_sceneObjectInstantiators[keyCode] = HostClass::Instantiate;
    }

private:
    enum JobIDs : uint { kJobDraw };

    JobManager                                  m_jobManager;

    AssetHandle<GI2D::Host::OverlayLayer>       m_overlayRenderer;
    AssetHandle<GI2D::Host::PathTracerLayer>    m_pathTracerLayer;
    //AssetHandle<GI2D::Host::IsosurfaceExplorer> m_isosurfaceExplorer;

    AssetHandle<GI2D::Host::SceneDescription>   m_scene;
    //std::vector<Cuda::BBox2f>                   m_tracableBBoxes;

    std::unique_ptr<GI2D::ViewTransform2D>      m_viewTransform;

    Cuda::AssetHandle<Cuda::RenderObjectContainer> m_renderObjects;

    GI2D::UIGridCtx                             m_gridCtx;
    GI2D::UIViewCtx                             m_viewCtx;
    GI2D::UISelectionCtx                        m_selectionCtx;

    std::unordered_map<uint, std::function<AssetHandle<GI2D::Host::SceneObjectInterface>(const std::string&)>>    m_sceneObjectInstantiators;

    struct
    {
        AssetHandle<GI2D::Host::SceneObjectInterface>   newObject;
    } 
    m_onCreate;

    struct
    {
        Cuda::vec2                              dragAnchor;
    }
    m_onMove;
};