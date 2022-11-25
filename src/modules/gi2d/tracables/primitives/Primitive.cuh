#pragma once

#include "../../Common.cuh"

#include "kernels/CudaManagedObject.cuh"
#include "kernels/CudaImage.cuh"

#include "../../Ray2D.cuh"

using namespace Cuda;

namespace Core
{
    namespace Host { template<typename T> class Vector; }
}

namespace GI2D
{   
    enum GI2DPrimitiveFlags : int
    {
        k2DPrimitiveSelected = 1
    };

    class Primitive2D
    {
    protected:
        __host__ __device__ Primitive2D() noexcept : m_flags(0) {}
        __host__ __device__ Primitive2D(const uchar& f, const vec3& c) noexcept : m_flags(f), m_colour(c) {}

        uchar m_flags;
        vec3 m_colour;

    public:
        __host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }
        __host__ __device__ virtual float                   Evaluate(const vec2& p, const float& dPdXY) const { return 0.0f; }
        __host__ __device__ virtual bool                    TestPoint(const vec2& p, const float& thickness) const { return false; }
        __host__ __device__ virtual bool                    IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const { return kFltMax; }
        __host__ __device__ virtual BBox2f                  GetBoundingBox() const { return BBox2f::MakeInvalid(); }
        __host__ __device__ virtual bool                    Intersects(const BBox2f& bBox) const { return false; }

        __host__ __device__ __forceinline__ bool            IsSelected() const { return m_flags & k2DPrimitiveSelected; }
        __host__ __device__ __forceinline__ void            SetFlags(const uchar flags, const bool set) 
        { 
            if (set) { m_flags |= flags; }
            else     { m_flags &= ~flags; }
        }
        __host__ __device__ __forceinline__ const vec3&     GetColour() const { return m_colour; }
        //__host__ __device__ __forceinline__ void            UnsetFlags(const uchar flags) { m_flags &= ~flags; }
    };
};