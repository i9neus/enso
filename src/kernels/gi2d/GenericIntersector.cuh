#include "Ray2D.cuh"

namespace GI2D
{
    __host__ __device__ __forceinline__ bool IntersectRayBBox(const RayBasic2D& ray, const BBox2f& bBox, vec2& t)
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

        t[0] = max(0.f, cwiseMax(tNearPlane));
        t[1] = cwiseMin(tFarPlane);
        return t[0] < t[1];
    }
}