#pragma once

#include "core/math/Math.cuh"
#include "core/math/Hash.cuh"

namespace Enso
{
    namespace Texture
    {
        __device__ float EvaluateFBM2D(vec2 uv, const float scale, const int numHarmonics)
        {
            uv = (uv + vec2(1e3f)) * scale;
            float accum = 0.;
            for (int harmIdx = 0; harmIdx < numHarmonics; ++harmIdx)
            {
                const vec2 uvHarm = uv * powf(2.f, float(harmIdx));
                const ivec2 ijHarm = ivec2(uvHarm);
                vec4 p;
                p.x = HashOfAsFloat(harmIdx, ijHarm.x, ijHarm.y);
                p.y = HashOfAsFloat(harmIdx, ijHarm.x + 1, ijHarm.y);
                p.z = HashOfAsFloat(harmIdx, ijHarm.x, ijHarm.y + 1);
                p.w = HashOfAsFloat(harmIdx, uvHarm.x + 1, ijHarm.y + 1);

                accum += smoothstep(smoothstep(p.x, p.y, fract(uvHarm.x)), smoothstep(p.z, p.w, fract(uvHarm.x)), fract(uvHarm.y)) / powf(2.f, float(1 + harmIdx));
            }
            return saturatef(accum);
        }

        __device__ float EvaluateGrid2D(const vec2& uv, const float thickness)
        {
            return step(thickness, fract(uv.x)) * step(thickness, fract(uv.y));
        }
    }

}