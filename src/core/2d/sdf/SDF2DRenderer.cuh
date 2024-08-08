#pragma once

#include "SDF2D.cuh"

namespace Enso
{
#define kInvalidSDF vec3(kFltMax)

    namespace SDF2DRenderer
    {
        // Render a line
        __host__ __device__ __forceinline__ float Line(const vec2& p, const vec2& v, const vec2& dv, const float& thickness, const float& dPdXY)
        {
            return saturatef(1.0f - (length(p - SDF2D::PerpLine(p, v, dv)) - dPdXY * thickness * 0.5f) / dPdXY);
        }

        // Render a hashed line
        __host__ __device__ __forceinline__ float HashedLine(const vec2& p, const vec2& v, const vec2& dv, const float& thickness, const float& hashFreq, const float& dPdXY)
        {
            return saturatef(1.0f - (length(p - SDF2D::PerpLineHashed(p, v, dv, hashFreq)) - dPdXY * thickness * 0.5f) / dPdXY);
        }

        // Render a filled ellipse
        __host__ __device__ __forceinline__ float Ellipse(const vec2& p, const vec2& origin, const float& radius, const float& dPdXY)
        {
            return saturatef(((radius + dPdXY) - length(p - origin)) / dPdXY);
        }

        // Render a torus
        __host__ __device__ __forceinline__ float Torus(const vec2& p, const vec2& origin, const float& radius, const float& thickness, const float& dPdXY)
        {
            return saturatef(((dPdXY * thickness * 0.5f) - fabsf(fabs(length(p - origin)) - radius)) / dPdXY);
        }

        // Render an arc
        __host__ __device__  float Arc(vec2 p, const vec2& o, const float& r, const vec2& range, const float& thickness, const float& dXYdP);

        // Render an activity indicator
        __host__ __device__ float ActivityIndicator(const vec2& uvView, const vec2& origin, const float radius, const float alpha, const float& dPdXY);
    }
}