#pragma once

#include "BBox2.cuh"

namespace Enso
{
    __host__ __device__ __forceinline__ BBox2f LineBBox2(const vec2& v0, const vec2& v1)
    {
        return BBox2f(vec2(fminf(v0.x, v1.x), fminf(v0.y, v1.y)),
                      vec2(fmaxf(v0.x, v1.x), fmaxf(v0.y, v1.y)));
    }

    __host__ __device__ __forceinline__ BBox2f CircleBBox2(const vec2& o, const float& r)
    {
        return BBox2f(o - vec2(r), o + vec2(r));
    }
}