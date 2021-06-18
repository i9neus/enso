#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"

namespace Cuda
{
    #define kNoIntersect -FLT_MAX

    namespace Intersector
    {
        // Ray-unit-box intersection test
        __host__ __device__ __forceinline__ float RayUnitBox(const RayBasic& ray)
        {
            vec3 tBackPlane, tFrontPlane;
            for (int dim = 0; dim < 3; dim++)
            {
                if (fabsf(ray.d[dim]) > 1e-10f)
                {
                    const float t0 = (0.5f - ray.o[dim]) / ray.d[dim];
                    const float t1 = (-0.5f - ray.o[dim]) / ray.d[dim];
                    if (t0 < t1) { tBackPlane[dim] = t0;  tFrontPlane[dim] = t1; }
                    else { tBackPlane[dim] = t1;  tFrontPlane[dim] = t0; }
                }
            }

            const float tBackMax = cwiseMax(tBackPlane);
            const float tFrontMin = cwiseMin(tFrontPlane);
            if (tBackMax > tFrontMin) { return kNoIntersect; }  // Ray didn't hit the box

            return max(0.0f, tBackMax);
        }

        __device__ __forceinline__ float RayPlane(const RayBasic& ray, const vec3& p, const vec3& n)
        {
            const float dDotN = dot(ray.d, n);
            return (dDotN < 1e-10f) ? kNoIntersect : ((dot(p, n) - dot(ray.o, n)) / dDotN);
        }
    }
}
