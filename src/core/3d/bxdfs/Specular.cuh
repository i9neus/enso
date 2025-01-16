#pragma once

#include "BxDF.cuh"
#include "../Ray.cuh"
#include "core/math/Mappings.cuh"
#include "core/3d/Basis.cuh"

namespace Enso
{
    namespace BxDF
    {
        constexpr float kMaxPDF = 1e6f;

        __device__ __forceinline__ float SamplePerfectSpecular(const vec3& i, const vec3& n, vec3& o)
        {
            o = Reflect(-i, n);
            return kMaxPDF;
        }

        __device__ bool SamplePerfectDielectric(const float& xi, const vec3& i, vec3 n, const float& ior, const bool isBackfacing, vec3& o, float& kickoff)
        {
            // Figure out what kind of intersection we're doing
            const vec2 eta = isBackfacing ? vec2(ior, 1.) : vec2(1., ior);           

            // Calculate the Fresnel coefficient and associated vectors. 
            float F = Fresnel(dot(i, n), eta.x, eta.y);
            if (xi > F)
            {
                o = Refract(-i, n, eta.x / eta.y);
                kickoff = -kickoff;
                return true;
            }
            else
            {
                o = Reflect(-i, n);
                return false;
            }
        }
    }
}