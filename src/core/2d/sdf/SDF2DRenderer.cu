#pragma once

#include "SDF2DRenderer.cuh"

namespace Enso
{
    namespace SDF2DRenderer
    {
        // Render an arc
        __host__ __device__ float Arc(vec2 p, const vec2& o, const float& r, const vec2& range, const float& thickness, const float& dXYdP)
        {
            p -= o;
            float phi = atan2f(p.y, p.x) + kPi;
            if ((range.x < range.y && (phi < range.x || phi > range.y)) ||
                (range.x > range.y && (phi > range.x || phi < range.y)))
            {
                phi = (fminf(fabsf(range.x - phi), fabsf(range.x - kTwoPi * sign(range.x - kPi) - phi)) <
                    fminf(fabsf(range.y - phi), fabsf(range.y - kTwoPi * sign(range.y - kPi) - phi))) ? range.x : range.y;
            }

            const vec2 pPerp = r * vec2(-cos(phi), sin(-phi));
            return saturatef((thickness * 0.5f - length(p - pPerp)) / dXYdP);
        }

        // Render an activity indicator
        __host__ __device__ float ActivityIndicator(const vec2& uvView, const vec2& origin, const float radius, float alpha, const float& dPdXY)
        {
            // Exclude anything outside the outer radius
            if (length2(uvView - origin) > sqr(radius * 1.15f))
            {
                return 0.f;
            }
            else
            {
                alpha = fract(alpha);
                
                // Define an arc that cyclically folds and unfolds
                const float phiStart = kTwoPi * fmaxf(0.f, 2.f * alpha - 1.f) - 1e-4f;
                const float phiEnd = kTwoPi * fminf(1.f, alpha * 2.f);

                return Torus(uvView, origin, radius, radius * 0.1f, dPdXY) +
                        Arc(uvView, origin, radius * 0.7f, vec2(phiStart, phiEnd), radius * 0.25f, dPdXY);
            }
        }
    }
}