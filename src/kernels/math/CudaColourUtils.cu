#pragma once

#include "CudaColourUtils.cuh"

namespace Cuda
{
    __host__ __device__  vec3 Hue(float phi)
    {
        phi *= 6.0f;
        const int i = int(phi);
        const vec3 c0 = vec3(((i + 4) / 3) & 1, ((i + 2) / 3) & 1, ((i + 0) / 3) & 1);
        const vec3 c1 = vec3(((i + 5) / 3) & 1, ((i + 3) / 3) & 1, ((i + 1) / 3) & 1);
        return mix(c0, c1, phi - float(i));
    }

    __host__ __device__ vec3 Heatmap(float phi)
    {
        constexpr int kGradLevels = 7;
        phi *= kGradLevels;
        int i = int(phi);
        if (phi >= kGradLevels) { phi = kGradLevels; i = kGradLevels - 1; }
        switch (i)
        {
        case 0: return mix(vec3(0.0f),             vec3(0.5f, 0.0f, 0.5f), phi - float(i)); 
        case 1: return mix(vec3(0.5f, 0.0f, 0.5f), vec3(0.0f, 0.0f, 1.0f), phi - float(i));
        case 2: return mix(vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 1.0f, 1.0f), phi - float(i));
        case 3: return mix(vec3(0.0f, 1.0f, 1.0f), vec3(0.0f, 1.0f, 0.0f), phi - float(i));
        case 4: return mix(vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 0.0f), phi - float(i));
        case 5: return mix(vec3(1.0f, 1.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), phi - float(i));
        case 6: return mix(vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f), phi - float(i));
        }

        return kZero;
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

    __host__ __device__ vec3 RGBToCIEXYZ(const vec3& rgb)
    {
        return mat3(0.4887180f, 0.3106803f, 0.2006017f,
                    0.1762044f, 0.8129847f, 0.0108109f,
                    0.0, 0.0102048f, 0.9897952f) * rgb;
    }

    __host__ __device__ vec3 CIEXYZToRGB(const vec3& xyz)
    {
        return mat3(2.3706743f, -0.9000405f, -0.4706338f,
                    -0.5138850f, 1.4253036f, 0.0885814f,
                    0.0052982f, -0.0146949f, 1.0093968f) * xyz;
    }

    __host__ __device__ vec3 RGBToChroma(const vec3& rgb)
    {
        // Chroma space is a linear transform whereby RGB space is rotated such that greyscale values
        // are colinear to the Y axis.
        return mat3(0.5773502691896258f, 0.5773502691896258f, 0.5773502691896258f,
                    -0.7071067811865475f, 0.7071067811865475f, 0.0f,
                    -0.4082482904638631f, -0.4082482904638631f, 0.816496580927726f) * rgb;
    }

    __host__ __device__ vec3 ChromaToRGB(const vec3& chr)
    {
        return mat3(0.5773502691896258f, -0.7071067811865475f, -0.4082482904638631f,
                    0.5773502691896258f, 0.7071067811865475f, -0.4082482904638631f,
                    0.5773502691896258f, 0.f, 0.81649658092772f) * chr;
    }

    __host__ __device__ vec3 XYZToxyY(const vec3& xyz)
    {
        float sum = xyz.x + xyz.y + xyz.z;
        if (fabsf(sum) < 1e-10f) { sum = 1e-10f; }
        return vec3(xyz.x / sum, xyz.y / sum, xyz.y);
    }

    __host__ __device__ vec3 xyYToXYZ(const vec3& xyy)
    {
        return vec3(xyy.x * xyy.z / xyy.y, 
                    xyy.z, 
                    (1.0f - xyy.x - xyy.y) * xyy.z / xyy.y);
    }

    __host__ __device__ vec4 Blend(vec4 lowerRgba, const vec3& upperRgb, const float& upperAlpha)
    {
        // Assume that RGB values are premultiplied so that when alpha-composited, they don't need to be renormalised
        lowerRgba = mix(lowerRgba, vec4(upperRgb, 1.0f), upperAlpha);
        lowerRgba.xyz /= max(1e-10f, lowerRgba.w);

        return lowerRgba;
    }

    __host__ __device__ vec4 Blend(vec4 lowerRgba, const vec4& upperRgba)
    {
        // Assume that RGB values are premultiplied so that when alpha-composited, they don't need to be renormalised
        lowerRgba = mix(lowerRgba, vec4(upperRgba.xyz, 1.0f), upperRgba.w);
        lowerRgba.xyz /= max(1e-10f, lowerRgba.w);

        return lowerRgba;
    }
}