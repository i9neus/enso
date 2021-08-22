#pragma once

#include "CudaColourUtils.cuh"

namespace Cuda
{
    __host__ __device__  vec3 Hue(float phi)
    {
        phi *= 6.0f;
        int i = int(phi);
        const vec3 c0 = vec3(((i + 4) / 3) & 1, ((i + 2) / 3) & 1, ((i + 0) / 3) & 1);
        const vec3 c1 = vec3(((i + 5) / 3) & 1, ((i + 3) / 3) & 1, ((i + 1) / 3) & 1);
        return mix(c0, c1, phi - float(i));
    }

    __host__ __device__  vec3 HSLToRGB(const vec3& hsl)
    {
        const vec3 hue = Hue(hsl.x);
        const vec3 saturation = mix(vec3(0.5f), hue, hsl.y);
        return (hsl.z <= 0.5) ? (saturation * hsl.z * 2.0f) : mix(saturation, vec3(1.0f), (hsl.z - 0.5f) * 2.0f);
    }

    __host__ __device__  vec3 HSVToRGB(const vec3& hsv)
    {
        return mix(vec3(0.0f), mix(vec3(1.0f), Hue(hsv.x), hsv.y), hsv.z);
    }

    __host__ __device__  vec3 RGBToHSV(const vec3& rgb)
    {
        // Value
        vec3 hsv;
        hsv.z = cwiseMax(rgb);

        // Saturation
        const float chroma = hsv.z - cwiseMin(rgb);
        hsv.y = (hsv.z < 1e-10f) ? 0.0f : (chroma / hsv.z);

        // Hue
        if (chroma < 1e-10f)        { hsv.x = 0.0f; }
        else if (hsv.z == rgb.x)    { hsv.x = (1.0f / 6.0f) * (rgb.y - rgb.z) / chroma; }
        else if (hsv.z == rgb.y)    { hsv.x = (1.0f / 6.0f) * (2.0f + (rgb.z - rgb.x) / chroma); }
        else                        { hsv.x = (1.0f / 6.0f) * (4.0f + (rgb.x - rgb.y) / chroma); }

        return hsv;
    }
}