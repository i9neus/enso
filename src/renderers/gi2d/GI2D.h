#pragma once

#include "../RendererInterface.h"
#include "generic/Job.h"

namespace Cuda
{
    struct GI2DOverlayParams;
    namespace Host
    {
        class GI2DOverlay;
    }
}

enum GI2DDirtyFlags : int
{
    kGI2DClean,
    kGI2DDirty
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
    virtual void OnDestroy() override final;
    virtual void OnPreRender() override final;
    virtual void OnRender() override final;
    virtual void OnPostRender() override final;

    Cuda::mat3 ConstructViewMatrix(const Cuda::vec2& trans, const float rotate, const float scale) const;

private:
    enum JobIDs : uint { kJobDraw };

    Cuda::AssetHandle<Cuda::Host::GI2DOverlay>  m_overlayRenderer;
    JobManager                                  m_jobManager;

    std::unique_ptr<Cuda::GI2DOverlayParams>    m_overlayParams;

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
    }
    m_view;
};