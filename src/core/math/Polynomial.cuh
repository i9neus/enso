#pragma once

#include "Math.cuh"

namespace Enso
{
    namespace Poly
    {
        struct Linear
        {
            // Find t for f(t) = 0
            __host__ __device__ __forceinline__ static float Solve(const vec2& s)
            {
                return -s.y / s.x;
            }

            // Evaulate f
            __host__ __device__ __forceinline__ static float Evaluate(const vec2& s, const float t)
            {
                return s.x + (s.y - s.x) * t;
            }
        };
        
        struct Quadratic
        {
            // Find f for f(t) = 0
            __host__ __device__ static bool Solve(const vec3& s, float& t0, float& t1)
            {
                float b2ac4 = s.y * s.y - 4.0 * s.x * s.z;
                if (b2ac4 < 0.0) { return false; }

                float sqrtb2ac4 = sqrt(b2ac4);
                t0 = (-s.y + sqrtb2ac4) / (2.0 * s.x);
                t1 = (-s.y - sqrtb2ac4) / (2.0 * s.x);
                return true;
            }

            // Evaluate f
            __host__ __device__ __forceinline__ static float Evaluate(const vec3& s, const float t)
            {
                return t * (s.x * t + s.y) + s.z;
            }

            // Evaluate df/dt
            __host__ __device__ __forceinline__ static float DEvaluate(const vec3& s, const float t)
            {
                return 2 * s.x * t + s.y;
            }

            // Solve for t where df/dt = 0
            __host__ __device__ __forceinline__ static float CriticalPoint(const vec3& s)
            {
                return -s.y / (2 * s.x);
            }
        };

        struct Cubic
        {
            // Returns the number of roots
            __host__ __device__ static int Solve(const vec4& s, vec3& r)
            {
                // Not a cubic equation, so try and solve as a quadtratic
                if (fabsf(s.x) < 1e-10f)
                {
                    const float c2bd4 = s.z * s.z - 4.0 * s.y * s.w;

                    // Not a quadratic equation either, so try and solve linearly
                    if (c2bd4 < 1e-10f)
                    {
                        if (fabsf(s.z) < 1e-10f) { return 0; } // No solutions

                        r[0] = -s.w / s.z;
                        return 1;
                    }

                    float sqrtc2bd4 = sqrt(c2bd4);
                    r[0] = (-s.z + sqrtc2bd4) / (2.0 * s.y);
                    r[1] = (-s.z - sqrtc2bd4) / (2.0 * s.y);
                    return 2;
                }

                // Re-express cubic in depressed form
                const float p = (3.0 * s.x * s.z - s.y * s.y) / (3.0 * s.x * s.x);
                const float q = (2.0 * s.y * s.y * s.y - 9.0 * s.x * s.y * s.z + 27.0 * s.x * s.x * s.w) / (27.0 * s.x * s.x * s.x);
                const float det = 4.0 * p * p * p + 27.0 * q * q;

                // Only one solution
                if (det > 0.0)
                {
                    const float alpha = sqrt(q * q / 4.0 + p * p * p / 27.0);
                    const float t = cubrt(-q / 2.0 + alpha) + cubrt(-q / 2.0 - alpha);

                    r[0] = t - s.y / (3.0 * s.x);
                    return 1;
                }
                // Three solutions
                else if (det < 0.0)
                {
                    const float alpha = acos(3.0 * q / (2.0 * p) * sqrt(-3.0 / p)) / 3.0;
                    const float beta = 2.0 * sqrt(-p / 3.0);
                    for (int i = 0; i < 3; i++)
                    {
                        float t = beta * cos(alpha - 2.0 * kPi * float(i) / 3.0);
                        r[i] = t - s.y / (3.0 * s.x);
                    }
                    return 3;
                }

                return 0;
            }

            // Evaluate f
            __host__ __device__ __forceinline__ static float Evaluate(const vec4& s, const float t)
            {
                return t * (t * (s.x*t + s.y) + s.z) + s.w;
            }

            // Evaluate df/dt
            __host__ __device__ __forceinline__ static float DEvaluate(const vec4& s, const float t)
            {
                return t * (3*s.x*t + 2*s.y) + s.z;
            }
        };

    };
}
