#pragma once

#include "CudaObjectManager.h"
#include "RenderManagerInterface.h"

class ShaderSandboxManager : public CudaObjectManager, public RenderManagerInterface
{
public:
    ShaderSandboxManager();

    virtual void Initialise() override final;
    virtual void Destroy() override final;

    virtual void OnResizeClient() override final;
};