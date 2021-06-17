#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"

namespace Cuda
{
    #define kNoIntersect -1.0f

    namespace Intersector
    {
        // Ray-unit-box intersection test
        __device__ __forceinline__ float RayUnitBox(const RayBasic& ray)
        {
            vec3 tNearPlane, tFarPlane;
            for (int dim = 0; dim < 3; dim++)
            {
                if (fabsf(ray.d[dim]) > 1e-10f)
                {
                    const float t0 = (0.5f - ray.o[dim]) / ray.d[dim];
                    const float t1 = (-0.5f - ray.o[dim]) / ray.d[dim];
                    if (t0 < t1) { tNearPlane[dim] = t0;  tFarPlane[dim] = t1; }
                    else { tNearPlane[dim] = t1;  tFarPlane[dim] = t0; }
                }
            }

            const float tNearMax = cwiseMax(tNearPlane);
            const float tFarMin = cwiseMin(tFarPlane);
            if (tNearMax > tFarMin) { return kNoIntersect; }  // Ray didn't hit the box

            if (tNearMax > 0.0) { return tNearMax; }
            else if (tFarMin > 0.0) { return tFarMin; }

            return 0.0f;  // Ray is inside box
        }

        __device__ __forceinline__ float RayPlane(const RayBasic& ray, const vec3& p, const vec3& n)
        {
            const float dDotN = dot(ray.d, n);
            return (dDotN < 1e-10f) ? kNoIntersect : ((dot(p, n) - dot(ray.o, n)) / dDotN);
        }
    }
}
