#pragma once

#include "core/math/Math.cuh"

namespace Enso
{
    __host__ __device__ __forceinline__ vec2 PerpLine(const vec2& p, const vec2& v0, const vec2& dv)
    {
        return v0 + saturatef((dot(p, dv) - dot(v0, dv)) / dot(dv, dv)) * dv;
    }

    __host__ __device__ __forceinline__ vec3 SDFLine(const vec2& p, const vec2& v0, const vec2& dv)
    {
        const vec2 perp = PerpLine(p, v0, dv);
        return vec3(length(perp - p), perp - p);
    }
}