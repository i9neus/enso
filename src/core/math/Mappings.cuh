#pragma once

#include "Math.cuh"

namespace Enso
{    
    // Quick and dirty method for sampling the unit disc from two canonical random variables. For a better algorithm, see
    // A Low Distortion Map Between Disk and Square (Shirley and Chiu)
    __host__ __device__ __forceinline__ vec2 SampleUnitDisc(const vec2& xi)
    {
        float phi = xi.y * kTwoPi;
        return vec2(sin(phi), cos(phi)) * sqrt(xi.x);
    }

    __host__ __device__ __forceinline__ vec2 SampleUnitDiscLowDistortion(const vec2& xi)
    {
        float phi, r;
        const float a = 2.0f * xi.x - 1.0f, b = 2.0f * xi.y - 1.0f;

        // From A Low Distortion Map Between Disk and Square (Shirley and Chiu)
        if (a > -b) // region 1 or 2
        {
            if (a > b) // region 1, also |a| > |b|
            {
                r = a;
                phi = (kPi / 4) * (b / a);
            }
            else // region 2, also |b| > |a|
            {
                r = b;
                phi = (kPi / 4) * (2 - (a / b));
            }
        }
        else // region 3 or 4
        {
            if (a < b) // region 3, also |a| >= |b|, a != 0
            {
                r = -a;
                phi = (kPi / 4) * (4 + (b / a));
            }
            else // region 4, |b| >= |a|, but a==0 and b==0 could occur.
            {
                r = -b;
                phi = (b != 0) ? ((kPi / 4) * (6 - (a / b))) : 0;
            }
        }

        return vec2(r * cosf(phi), r * sinf(phi));
    }

    __host__ __device__ __forceinline__ vec3 SampleUnitSphere(vec2 xi)
    {
        xi.x = xi.x * 2.0 - 1.0;
        xi.y *= kTwoPi;

        float sinTheta = sqrt(1.0 - xi.x * xi.x);
        return vec3(cos(xi.y) * sinTheta, xi.x, sin(xi.y) * sinTheta);
    }

    __host__ __device__ __forceinline__ vec3 SampleUnitHemisphere(vec2 xi)
    {
        xi.y *= kTwoPi;

        float sinTheta = sqrt(1.0 - xi.x * xi.x);
        return vec3(cos(xi.y) * sinTheta, sin(xi.y) * sinTheta, xi.x);
    }

    // Converts a normalised direction into normalised lat-long coordinates (used by HDRI probes)
    __host__ __device__ __forceinline__ vec2 DirToEquirect(const vec3& n)
    {
        return vec2((atan2f(n.z, n.x) + kPi) / kTwoPi, acosf(clamp(n.y, -1.f, 1.f)) / kPi);
    }
}