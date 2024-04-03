#pragma once

#include "core/math/Math.cuh"

namespace Enso
{
    #define kInvalidSDF vec3(kFltMax)
    
    __host__ __device__ __forceinline__ vec2 PerpLine(const vec2& p, const vec2& v0, const vec2& dv)
    {
        return v0 + saturatef((dot(p, dv) - dot(v0, dv)) / dot(dv, dv)) * dv;
    }

    __host__ __device__ __forceinline__ vec3 SDFLine(const vec2& p, const vec2& v0, const vec2& dv)
    {
        const vec2 perp = PerpLine(p, v0, dv);
        return vec3(length(perp - p), p - perp);
    }

    __host__ __device__ __forceinline__ vec2 PerpPoint(const vec2& p, const vec2& v0, const float r)
    {
        return v0 + normalize(p - v0);
    }

    __host__ __device__ __forceinline__ vec3 SDFPoint(const vec2& p, const vec2& v0, const float r)
    {
        const float dist = length(p - v0);
        return vec3(dist, (p - v0) / dist);
    }
}