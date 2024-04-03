#pragma once

#include "../Common.cuh"

#include "core/Image.cuh"

#include "../Ray2D.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }

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
        __host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { dassert("PerpendicularPoint not implemented."); return vec2(0.0f); }
        __host__ __device__ virtual float                   EvaluateOverlay(const vec2& p, const float& dPdXY) const { dassert("EvaluateOverlay not implemented."); return 0.0f; }
        __host__ __device__ virtual bool                    TestPoint(const vec2& p, const float& thickness) const { dassert("TestPoint not implemented."); return false; }
        __host__ __device__ virtual bool                    IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const { dassert("IntersectRay not implemented."); return kFltMax; }
        __host__ __device__ virtual BBox2f                  GetBoundingBox() const { dassert("GetBoundingBox not implemented."); return BBox2f::MakeInvalid(); }
        __host__ __device__ virtual bool                    Intersects(const BBox2f& bBox) const { dassert("Intersects not implemented."); return false; }
        __host__ __device__ virtual bool                    Contains(const vec2& p, const float& dPdXY) const { dassert("Contains not implemented."); return false; }

        __host__ __device__ __forceinline__ bool            IsSelected() const { return m_flags & k2DPrimitiveSelected; }
        __host__ __device__ __forceinline__ void            SetFlags(const uchar flags, const bool set)
        {
            if (set) { m_flags |= flags; }
            else { m_flags &= ~flags; }
        }
        __host__ __device__ __forceinline__ const vec3& GetColour() const { return m_colour; }
        //__host__ __device__ __forceinline__ void            UnsetFlags(const uchar flags) { m_flags &= ~flags; }
    };
}