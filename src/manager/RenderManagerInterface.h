#pragma once

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/HighResolutionTimer.h"
#include "generic/Math.h"
#include <cuda_runtime.h>
#include "generic/JsonUtils.h"
#include <deque>

class RenderManagerInterface
{
public:
    virtual void Initialise() = 0;
    virtual void Destroy() = 0;

    virtual void OnResizeClient() = 0;

protected:
    RenderManagerInterface();
    virtual ~RenderManagerInterface();
};