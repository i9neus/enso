#pragma once

#include "core/math/Math.cuh"
#include "core/math/Hash.cuh"

namespace Enso
{
    namespace Device
    {
        __device__ float FBM2D(vec2 uv, const float& scale, const int& numHarmonics)
        {
            uv = (uv + vec2(1e3f)) * scale;
            float accum = 0.;
            for (int harmIdx = 0; harmIdx < numHarmonics; ++harmIdx)
            {
                const vec2 uvHarm = uv * powf(2.f, float(harmIdx));
                vec4 p;
                p.x = HashOfAsFloat(harmIdx, uvHarm.x, uvHarm.y);
                p.y = HashOfAsFloat(harmIdx, uvHarm.x + 1, uvHarm.y);
                p.z = HashOfAsFloat(harmIdx, uvHarm.x, uvHarm.y + 1);
                p.w = HashOfAsFloat(harmIdx, uvHarm.x + 1, uvHarm.y + 1);

                accum += smoothstep(smoothstep(p.x, p.y, fract(uvHarm.x)), smoothstep(p.z, p.w, fract(uvHarm.x)), fract(uvHarm.y)) / powf(2.f, float(1 + harmIdx));
            }
            return saturatef(accum);
        }
    }
}