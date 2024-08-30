#pragma once

#include "core/math/Math.cuh"
#include "core/math/samplers/PCG.cuh"
#include "core/math/samplers/owensobol/OwenSobol.cuh"

namespace Enso
{
    struct RenderCtx
    {
        PCG             rng;
        OwenSobol       qrng;

        int             frameIdx;
        struct
        {
            ivec2 xy;
            ivec2 dims;
        } 
        viewport;

        __device__ __forceinline__ vec4 Rand(const uint dim) 
        { 
            //return (viewport.xy.x < viewport.dims.x / 2) ? rng.Rand() : qrng.Rand(dim);
            return qrng.Rand(dim);
        }
    };
    
    struct HitCtx
    {
        vec3            n;
        vec2            uv;
        vec3            albedo;
    };
}