#pragma once

#include "RendererInterface.h"

namespace Cuda
{
    class TestCard;
}

class GI2D : public RendererInterface
{
public:
    GI2D(); 
    virtual ~GI2D();

    virtual void Initialise() override final;

    virtual void OnMouseMove(const int clientX, const int clientY) override final;
    virtual void OnMouseButton(const int code, const bool isDown) override final;
    virtual void OnMouseWheel(const float degrees) override final;
    virtual void OnKey(const int code, const bool isDown) override final;

    virtual void OnResizeClient() override final;
    virtual std::string GetRendererName() const { return "2D GI Sandbox"; };

    static std::shared_ptr<RendererInterface> Instantiate();

protected:


private:
    virtual void OnDestroy() override final;
    virtual void OnPreRender() override final;
    virtual void OnRender() override final;
    virtual void OnPostRender() override final;

private:
    Cuda::AssetHandle<Cuda::TestCard>   m_hostTestCard;
};