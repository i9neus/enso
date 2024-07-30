#pragma once

#include "core/math/Math.cuh"

namespace Enso
{
    __host__ __device__ __forceinline__ mat3 CreateBasis(const vec3& n)
    {
        const float s = sign(n.z);
        const float a = -1.0f / (s + n.z);
        const float b = n.x * n.y * a;

        return mat3(vec3(1.0f + s * n.x * n.x * a, s * b, -s * n.x),
                    vec3(b, s + n.y * n.y * a, -n.y),
                    n);
    }

    __host__ __device__ __forceinline__ mat3 CreateBasis(const vec3& n, const vec3& up)
    {
        const vec3 tangent = normalize(cross(n, up));  
        return transpose(mat3(tangent, 
                              cross(tangent, n), 
                              n));
    }
}