#include "CudaPrimitive2D.cuh"
#include "../CudaVector.cuh"

namespace Cuda
{
    __host__ __device__ __forceinline__ vec2 LineSegment::PerpendicularPoint(const vec2& p) const
    {
       return v + saturate((dot(p, dv) - dot(v, dv)) / dot(dv, dv)) * dv;
    }

    __host__ __device__ float LineSegment::Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const
    {
        return saturate(1.0f - (length(p - PerpendicularPoint(p)) - thickness) / dPdXY);
    }

    __host__ __device__ bool LineSegment::Intersects(const vec2& p, const float& thickness) const
    {
        return length2(p - PerpendicularPoint(p)) < sqr(thickness);
    }
}