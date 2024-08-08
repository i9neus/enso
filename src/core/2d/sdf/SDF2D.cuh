#pragma once

#include "core/math/Math.cuh"

namespace Enso
{
    #define kInvalidSDF vec3(kFltMax)

    namespace SDF2D
    {
        /*
        * SDF functions should return vec3 triples:
            x: the distance to the zero point of the field
            yz: the closest position on the zero point of the field
        */

        // Penpendicular distance to a line
        __host__ __device__ __forceinline__ vec2 PerpLine(const vec2& p, const vec2& v0, const vec2& dv)
        {
            return v0 + saturatef((dot(p, dv) - dot(v0, dv)) / dot(dv, dv)) * dv;
        }

        // Penpendicular distance to a hashed line
        __host__ __device__ __forceinline__ vec2 PerpLineHashed(const vec2& p, const vec2& v0, const vec2& dv, const float hashFreq)
        {
            float t = saturatef((dot(p, dv) - dot(v0, dv)) / dot(dv, dv)) * hashFreq;
            float f = fract(t);
            if (f < 0.5f) { t -= f; }

            return v0 + (t / hashFreq) * dv;
        }

        // SDF for a line
        __host__ __device__ __forceinline__ vec3 Line(const vec2& p, const vec2& v0, const vec2& dv)
        {
            const vec2 perp = PerpLine(p, v0, dv);
            return vec3(length(perp - p), p - perp);
        }

        // Perpendicular distance to a point
        __host__ __device__ __forceinline__ vec2 PerpPoint(const vec2& p, const vec2& v0, const float r)
        {
            return v0 + normalize(p - v0);
        }

        // SDF for a point
        __host__ __device__ __forceinline__ vec3 Point(const vec2& p, const vec2& v0, const float r)
        {
            const float dist = length(p - v0);
            return vec3(dist, (p - v0) / dist);
        }            
    }  
}