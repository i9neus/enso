#pragma once

#include "../Ray2D.cuh"

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

        range.tNear = cwiseMax(tNearPlane);
        range.tFar = cwiseMin(tFarPlane);
        return range.tNear <= range.tFar;
    }

    // Find the closest intersection between a ray and box. Returns -kFltMax if not hit and 0.0 if the ray origin is inside the box.
    __host__ __device__ __forceinline__ float IntersectRayBBox(const RayBasic2D& ray, const BBox2f& bBox)
    {
        RayRange2D range;
        if(!IntersectRayBBox(ray, bBox, range) || range.tFar < 0.0f) { return -kFltMax; }

        return (range.tNear < 0.0f) ? 0.0f : range.tNear;
    }

    __host__ __device__ __forceinline__ bool IntersectRayCircle(const RayBasic2D& ray, const vec2& origin, const float radius, RayRange2D& range)
    {
        // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
        vec2 o = ray.o - origin;
        float a = dot(ray.d, ray.d);
        float b = 2.0 * dot(ray.d, o);
        float c = dot(o, o) - sqr(radius);

        float b2ac4 = b * b - 4.0 * a * c;
        if (b2ac4 < 0.0) { return false; }

        float sqrtb2ac4 = sqrt(b2ac4);
        float t0 = (-b + sqrtb2ac4) / (2.0 * a);
        float t1 = (-b - sqrtb2ac4) / (2.0 * a);

        range = (t1 > t0) ? RayRange2D(t0, t1) : RayRange2D(t1, t0); 
        return range.tNear <= range.tFar;
    }
}