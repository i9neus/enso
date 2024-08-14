#pragma once

#include "Math.cuh"

namespace Enso
{
    struct Quaternion : public vec4
    {
        __host__ __device__ Quaternion() : vec4(0.0f) {}
        __host__ __device__ Quaternion(const float u, const float v, const float w, const float r) : vec4(u, v, w, r) {}
        __host__ __device__ Quaternion(const vec4& v) : vec4(v) {}

        __host__ __device__ __forceinline__ Quaternion& operator =(const vec4& rhs)
        {
            static_cast<vec4&>(*this) = rhs;
            return *this;
        }

        __host__ __device__ __forceinline__ mat3 RotationMatrix() const
        {
            return 2.f * mat3(vec3(0.5f - y * y - z * z, x * y + w * z, x * z - w * y),
                              vec3(x * y - w * z, 0.5f - x * x - z * z, y * z + w * x),
                              vec3(x * z + w * y, y * z - w * x, 0.5f - x * x - y * y));
        }
    };       
}