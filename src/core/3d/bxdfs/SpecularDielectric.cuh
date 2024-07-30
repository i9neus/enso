#pragma once

#include "../Ray.cuh"
#include "core/math/Mappings.cuh"
#include "core/3d/Basis.cuh"

namespace Enso
{
    namespace BxDF
    {
        __host__ __device__ void SampleLambert(const vec2& xi, const vec3& n, vec3& o, float& pdf)
        {
            // Sample the Lambertian direction
            vec3 r = vec3(SampleUnitDisc(xi), 0.0f);
            r.z = sqrt(1.0 - sqr(r.x) - sqr(r.y));

            // Transform it to world space
            o = CreateBasis(n) * r;
            pdf = r.z / kPi;
        }

        __host__ __device__ __forceinline__ float EvalauateLambert(const vec3& d, const vec3& n)
        {
            return dot(d, n) / kPi;
        }
    }
}