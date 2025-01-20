#pragma once

#include "core/math/Math.cuh"
#include "core/math/Polynomial.cuh"

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

        // Calculates the closest point on the quadratic curve to position xy
        __host__ __device__ static bool QuadradicSplinePerpendicularPoint(const vec2& p, const vec3& abcX, const vec3& abcY, const float margin, float& tPerp, vec2& xyPerp)
        {
            float a0 = abcX.x, b0 = abcX.y, c0 = abcX.z;
            float d0 = abcY.x, e0 = abcY.y, f0 = abcY.z;
            float n0 = p.x, m0 = p.y;

            float a = -2.0 * a0 * a0 - 2.0 * d0 * d0;
            float b = -3.0 * a0 * b0 - 3.0 * d0 * e0;
            float c = -b0 * b0 - 2.0 * a0 * c0 - e0 * e0 - 2.0 * d0 * f0 + 2.0 * d0 * m0 + 2.0 * a0 * n0;
            float d = -b0 * c0 - e0 * f0 + e0 * m0 + b0 * n0;

            vec3 solutions;
            const int numSolutions = Poly::Cubic::Solve(vec4(a, b, c, d), solutions);

            if (numSolutions == 0) { return false; }

            xyPerp = vec2(kFltMax);
            float nearest = kFltMax;
            for (int idx = 0; idx < numSolutions; ++idx)
            {
                float t = clamp(solutions[idx], -margin, 1.0 + margin);
                vec2 perp = vec2(abcX.x * t * t + abcX.y * t + abcX.z,
                    abcY.x * t * t + abcY.y * t + abcY.z);

                float dist = length2(p - perp);
                if (dist < nearest)
                {
                    nearest = dist;
                    xyPerp = perp;
                    tPerp = t;
                }
            }

            return true;
        }

        __host__ __device__ static bool QuadradicSplinePerpPointApprox(const vec2& p, const vec3& abcX, const vec3& abcY, const float margin, float& tPerp, vec2& xyPerp)
        {
            float a0 = abcX.x, b0 = abcX.y, c0 = abcX.z;
            float d0 = abcY.x, e0 = abcY.y, f0 = abcY.z;
            float n0 = p.x, m0 = p.y;

            const vec4 P(-2.0 * a0 * a0 - 2.0 * d0 * d0,
                         -3.0 * a0 * b0 - 3.0 * d0 * e0,
                         -b0 * b0 - 2.0 * a0 * c0 - e0 * e0 - 2.0 * d0 * f0 + 2.0 * d0 * m0 + 2.0 * a0 * n0,
                         -b0 * c0 - e0 * f0 + e0 * m0 + b0 * n0);

            vec3 t, f;
            float b2ac4 = sqr(2. * P.y) - 4.0 * (3. * P.x) * P.z;
            if (b2ac4 <= 1e-3 || fabsf(P.x) / fmaxf(1e-10, fabsf(P.y)) < 1.)
            {
                t.xz = vec2(-margin, 1. + margin);
                t.y = mix(t.x, t.z, 0.5);
            }
            else
            {
                b2ac4 = sqrt(b2ac4);
                float t0 = (-(2. * P.y) + b2ac4) / (2.0 * (3. * P.x));
                float t1 = (-(2. * P.y) - b2ac4) / (2.0 * (3. * P.x));

                t.x = t0 - (t1 - t0) * 0.5;
                t.y = (t0 + t1) * 0.5;
                t.z = t1 + (t1 - t0) * 0.5;
            }

#define Cubic(P, t) (P.w + (t) * (P.z + (t) * (P.y + (t) * P.x)))
#define DCubic(P, t) (P.z + (t) * (2. * P.y + (t) * 3. * P.x))

            constexpr int kNewtonIters = 5;
            for (int i = 0; i < kNewtonIters; ++i)
            {
                f = Cubic(P, t);
                t -= f / DCubic(P, t);
            }
            f = Cubic(P, t);

#undef Cubic
#undef DCubic

            xyPerp = vec2(kFltMax);
            float nearest = kFltMax;
            for (int idx = 0; idx < 3; ++idx)
            {
                if (fabsf(f[idx]) < 1e-5)
                {
                    const float td = clamp(t[idx], -margin, 1.0 + margin);
                    vec2 perp = vec2(abcX.x * td * td + abcX.y * td + abcX.z,
                                     abcY.x * td * td + abcY.y * td + abcY.z);
             
                    const float dist = length2(p - perp);
                    if (dist < nearest)
                    {
                        nearest = dist;
                        xyPerp = perp;
                        tPerp = td;
                    }
                }
            }

            return true;
        }

        // SDF for a quadratic spline
        __host__ __device__ __forceinline__ vec3 QuadradicSpline(const vec2& p, const vec3& abcX, const vec3& abcY)
        {
            vec2 pPerp;
            float tPerp;
            return QuadradicSplinePerpPointApprox(p, abcX, abcY, 0., tPerp, pPerp) ? vec3(length(pPerp - p), p - pPerp) : vec3(0.);
        }
    }  
}