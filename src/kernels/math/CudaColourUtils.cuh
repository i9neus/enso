#pragma once

#include "CudaMath.cuh"

namespace Cuda
{
    __device__ __forceinline__ vec3 Hue(float phi)
    {
        phi *= 6.0f;
        int i = int(phi);
        const vec3 c0 = vec3(((i + 4) / 3) & 1, ((i + 2) / 3) & 1, ((i + 0) / 3) & 1);
        const vec3 c1 = vec3(((i + 5) / 3) & 1, ((i + 3) / 3) & 1, ((i + 1) / 3) & 1);
        return mix(c0, c1, phi - float(i));
    }

    __device__ __forceinline__ vec3 HSL(vec3 hsl)
    {
        const vec3 hue = Hue(hsl.x);
        const vec3 saturation = mix(vec3(0.5f), hue, hsl.y);
        return (hsl.z <= 0.5) ? (saturation * hsl.z * 2.0f) : mix(saturation, vec3(1.0f), (hsl.z - 0.5f) * 2.0f);
    }
}