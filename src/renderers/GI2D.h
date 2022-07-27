#pragma once

#include "RendererInterface.h"

class GI2D : public RendererInterface
{
public:
    GI2D(); 
    virtual ~GI2D();

    virtual void Initialise() override final;
    virtual void Destroy() override final;

    virtual void OnResizeClient() override final;
    virtual const std::string& GetRendererName() const { return "2D GI Sandbox"; };

    static std::shared_ptr<RendererInterface> Instantiate();

protected:


private:
    virtual void PreRender() override final;
    virtual void Render() override final;
    virtual void PostRender() override final;
};