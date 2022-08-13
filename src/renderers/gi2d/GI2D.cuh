#pragma once

#include "../RendererInterface.h"
#include "generic/Job.h"

namespace Cuda
{
    struct GI2DOverlayParams; 
    struct LineSegment;
    class Asset;
    namespace Host
    {
        class BIH2DAsset;
        class GI2DOverlay;
        template<typename T> class Vector;
    }
}

struct CudaObjects;

enum GI2DDirtyFlags : int
{
    kGI2DClean = 0,
    kGI2DDirtyParams = 1,
    kGI2DDirtyLineSegments = 2
};

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

class GI2D : public RendererInterface
{
public:
    GI2D(); 
    virtual ~GI2D();

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

    void                    RebuildBIH();

    Cuda::mat3              ConstructViewMatrix(const Cuda::vec2& trans, const float rotate, const float scale) const;

    uint OnCreatePath(const UIStateTransition& transition);
    uint OnSelectPath(const UIStateTransition& transition);
    uint OnIdleState(const UIStateTransition& transition);

private:
    enum JobIDs : uint { kJobDraw };

    std::unique_ptr<CudaObjects>                m_objectsPtr;
    CudaObjects&                                m_objects;
    JobManager                                  m_jobManager;

    struct EditMode
    {
        uint                                    type;
        uint                                    stage;
    };
    EditMode                                    m_uiEditMode;
    EditMode                                    m_rendererEditMode;

    struct
    {
        Cuda::vec2                              trans;
        float                                   scale;
        float                                   rotate;
        float                                   zoomSpeed;

        Cuda::vec2                              dragAnchor;
        Cuda::vec2                              rotAxis;
        Cuda::vec2                              transAnchor;
        float                                   rotAnchor;
        float                                   scaleAnchor;

        Cuda::vec2                              mousePosView;
        //Cuda::mat3                              matrix;               <------- Using OverlayParams.viewMatrix instead
    }
    m_view;

    struct
    {
        uint                                    pathStartIdx;
        uint                                    numVertices;
        Cuda::vec2                              startPos;
    } 
    m_newPath;
};