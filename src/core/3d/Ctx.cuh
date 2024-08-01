#pragma once

#include "core/math/Math.cuh"
#include "core/math/samplers/PCG.cuh"

namespace Enso
{
    struct RenderCtx
    {
        PCG             rng;
        ivec2           xyScreen;
        int             frameIdx;

        __host__ __device__ __forceinline__ vec4 Rand() { return rng.Rand(); }
    };
    
    struct HitCtx
    {
        int             matID;
        vec3            n;
        vec2            uv;
        float           alpha;
    };
}