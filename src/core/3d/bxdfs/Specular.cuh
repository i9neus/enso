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

        __device__ bool SamplePerfectDielectric(const float& xi, const vec3& i, vec3 n, const float& ior, vec3& o, vec3& kickoff)
        {
            // Figure out what kind of intersection we're doing
            vec2 eta;
            if (dot(i, n) > 0.0)
            {
                eta = vec2(1., ior);
            }
            else
            {
                eta = vec2(ior, 1.);
                n = -n;
            }

            // Calculate the Fresnel coefficient and associated vectors. 
            float F = Fresnel(dot(i, n), eta.x, eta.y);
            if (xi > F)
            {
                o = Refract(-i, n, eta.x / eta.y);
                kickoff = -n * 1e-4;
                return true;
            }
            else
            {
                o = Reflect(-i, n);
                kickoff = n * 1e-4;
                return false;
            }
        }
    }
}