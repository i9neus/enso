#pragma once

#include "../Ray.cuh"

namespace Enso
{
    namespace Intersector
    {
        // Ray-plane intersection test
        __host__ __device__ __forceinline__ float RayPlane(const RayBasic& ray)
        {
            return (fabsf(ray.d.z) < 1e-15f) ? -kFltMax : (ray.o.z / -ray.d.z);
        }

        // Ray-sphere intersection test
        __host__ __device__ bool RayUnitSphere(const RayBasic& ray, vec2& t)
        {
            const float a = dot(ray.d, ray.d);
            const float b = 2.0f * dot(ray.d, ray.o);
            const float c = dot(ray.o, ray.o) - 1.0;

            float b2ac4 = b * b - 4.0f * a * c;
            if (b2ac4 >= 0.0)
            {
                b2ac4 = sqrt(b2ac4);
                t.x = (-b + b2ac4) / (2.0f * a);
                t.y = (-b - b2ac4) / (2.0f * a);
                return true;
            }

            return false;
        }
    }
}