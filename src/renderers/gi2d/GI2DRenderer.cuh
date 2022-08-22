#pragma once

#include "../RendererInterface.h"
#include "generic/Job.h"
#include "kernels/gi2d/UICtx.cuh"
#include "kernels/CudaVector.cuh"

namespace GI2D
{
    struct OverlayParams;  
    struct PathTracerParams; 
    struct ViewTransform2D;
    class LineSegment; 

    namespace Device
    {
        class Tracable;
    }

    namespace Host
    {
        class BIH2DAsset;
        class Tracable;
        class Overlay;   
        class PathTracer; 
        class Tracable;
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
    //virtual void          OnPreRender() override final;
    virtual void            OnRender() override final;
    //virtual void          OnPostRender() override final;
    void                    OnViewChange();

    void                    Rebuild();

    uint                    OnMoveTracable(const uint& sourceStateIdx, const uint& targetStateIdx);
    uint                    OnCreateTracable(const uint& sourceStateIdx, const uint& targetStateIdx);
    uint                    OnSelectTracables(const uint& sourceStateIdx, const uint& targetStateIdx);
    uint                    OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx);
    uint                    OnDeletePath(const uint& sourceStateIdx, const uint& targetStateIdx);

    std::string             DecideOnClickState(const uint& sourceStateIdx);

private:
    enum JobIDs : uint { kJobDraw };

    JobManager                                  m_jobManager;

    AssetHandle<GI2D::Host::Overlay>            m_overlayRenderer;
    AssetHandle<GI2D::Host::PathTracer>         m_pathTracer;

    std::unique_ptr<GI2D::OverlayParams>        m_overlayParams;
    std::unique_ptr<GI2D::PathTracerParams>     m_pathTracerParams;
    AssetHandle<GI2D::Host::BIH2DAsset>         m_sceneBIH;
    AssetHandle<Cuda::Host::AssetVector<GI2D::Host::Tracable>> m_hostTracables;
    std::vector<Cuda::BBox2f>                   m_tracableBBoxes;

    std::unique_ptr<GI2D::ViewTransform2D>      m_viewTransform;

    Cuda::AssetHandle<Cuda::RenderObjectContainer> m_renderObjects;

    GI2D::UIViewCtx                             m_viewCtx;
    GI2D::UISelectionCtx                        m_selectCtx;

    struct
    {
        AssetHandle<GI2D::Host::Tracable>   newObject;
    } 
    m_createObject;

    struct
    {
        Cuda::vec2                              dragAnchor;
    }
    m_moveTracable;
};