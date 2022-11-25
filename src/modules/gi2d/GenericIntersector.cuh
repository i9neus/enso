#pragma once

#include "Ray2D.cuh"

namespace Enso
{
    __host__ __device__ __forceinline__ bool IntersectRayBBox(const RayBasic2D& ray, const BBox2f& bBox, RayRange2D& range)
    {
        vec2 tNearPlane, tFarPlane;
        for (int dim = 0; dim < 2; dim++)
        {
            if (fabs(ray.d[dim]) > 1e-10f)
            {
                float t0 = (bBox.upper[dim] - ray.o[dim]) / ray.d[dim];
                float t1 = (bBox.lower[dim] - ray.o[dim]) / ray.d[dim];
                if (t0 < t1) { tNearPlane[dim] = t0;  tFarPlane[dim] = t1; }
                else { tNearPlane[dim] = t1;  tFarPlane[dim] = t0; }
            }
        }

        range.tNear = fmaxf(0.f, cwiseMax(tNearPlane));
        range.tFar = cwiseMin(tFarPlane);
        return range.tNear < range.tFar;
    }
}