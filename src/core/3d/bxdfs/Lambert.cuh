#pragma once

#include "../Ray.cuh"
#include "core/math/Mappings.cuh"
#include "core/3d/Basis.cuh"

namespace Enso
{
    namespace BxDF
    {
        __host__ __device__ float SampleLambertian(const vec2& xi, const vec3& n, vec3& o)
        {
            // Sample the Lambertian direction
            vec3 r = vec3(SampleUnitDisc(xi), 0.0f);
            r.z = sqrt(1.0 - sqr(r.x) - sqr(r.y));

            // Transform it to world space
            o = CreateBasis(n) * r;
            return r.z / kPi;
        }

        __host__ __device__ __forceinline__ float EvaluateLambertian()
        {
            return 1.f / kPi;
        }
    }
}