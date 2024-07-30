#pragma once

#include "core/math/Math.cuh"
#include "core/math/samplers/PCG.cuh"

namespace Enso
{
    struct RenderCtx
    {
        PCG             rng;

        __host__ __device__ __forceinline__ vec4 Rand() { return rng.Rand(); }
    };
    
    struct HitCtx
    {
        int             hitId;
        vec3            n;
        vec2            uv;
    };
}