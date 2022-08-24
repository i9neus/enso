#pragma once

#include "Common.cuh"

#include "generic/StdIncludes.h"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"

#include "Ray2D.cuh"

using namespace Cuda;

namespace Cuda
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
        __host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const = 0;
        __host__ __device__ virtual float                   Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const = 0;
        __host__ __device__ virtual bool                    TestPoint(const vec2& p, const float& thickness) const = 0;
        __host__ __device__ virtual bool                    IntersectRay(const Ray2D& ray, HitCtx2D& hit) const = 0;
        __host__ __device__ virtual BBox2f                  GetBoundingBox() const = 0;
        __host__ __device__ virtual bool                    Intersects(const BBox2f& bBox) const = 0;

        __host__ __device__ __forceinline__ bool            IsSelected() const { return m_flags & k2DPrimitiveSelected; }
        __host__ __device__ __forceinline__ void            SetFlags(const uchar flags, const bool set) 
        { 
            if (set) { m_flags |= flags; }
            else     { m_flags &= ~flags; }
        }
        __host__ __device__ __forceinline__ const vec3&     GetColour() const { return m_colour; }
        //__host__ __device__ __forceinline__ void            UnsetFlags(const uchar flags) { m_flags &= ~flags; }
    };
    
    class LineSegment : public Primitive2D
    {
    private:
        vec2 m_v[2];
        vec2 m_dv;
    public:
        __host__ __device__ LineSegment() noexcept : Primitive2D(), m_v{ vec2(0.0f), vec2(0.0f) }, m_dv(0.0f) {}
        __host__ __device__ LineSegment(const vec2& v0, const vec2& v1, const uchar flags, const vec3& col) noexcept :
            Primitive2D(flags, col), m_v{ v0, v1 }, m_dv(v1 - v0) {}

        __host__ __device__  virtual vec2                   PerpendicularPoint(const vec2& p) const override final;
        __host__ __device__ virtual float                   Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const override final;
        __host__ __device__ virtual bool                    TestPoint(const vec2& p, const float& thickness) const override final;
        __host__ __device__ virtual bool                    IntersectRay(const Ray2D& ray, HitCtx2D& hit) const override final;
        __host__ __device__ virtual bool                    Intersects(const BBox2f& bBox) const override final;

        __host__ __device__ vec2                            PointAt(const float& t) const { return m_v[0] + m_dv * t; }
        
        __host__ __device__ __forceinline__ virtual BBox2f GetBoundingBox() const override final
        {
            return BBox2f(vec2(min(m_v[0].x, m_v[1].x), min(m_v[0].y, m_v[1].y)),
                          vec2(max(m_v[0].x, m_v[1].x), max(m_v[0].y, m_v[1].y)));
        }
        
        __host__ __device__ void Set(const uint& idx, const vec2& v)
        {
            m_v[idx] = v;
            m_dv = m_v[1] - m_v[0];
        }

        __host__ __device__ __forceinline__ LineSegment& operator+=(const vec2& v) 
        { 
            m_v[0] += v; m_v[1] += v; m_dv = m_v[1] - m_v[0];
            return *this;
        }

        __host__ __device__ __forceinline__ LineSegment& operator-=(const vec2& v)
        {
            m_v[0] -= v; m_v[1] -= v; m_dv = m_v[1] - m_v[0];
            return *this;
        }
        
        __host__ __device__ __forceinline__ const vec2& operator[](const uint& idx) const { return m_v[idx]; }
        __host__ __device__ __forceinline__ vec2& dv(const uint& idx) { return m_dv; }
    };

    __host__ void GenerateRandomLineSegments(Cuda::Host::Vector<LineSegment>& segments, const BBox2f& bounds, const ivec2 numSegmentsRange, const vec2 sizeRange, const uint seed);
    
    /*class Primitive2DContainer
    {
    public:
        __host__ Primitive2DContainer();
        
        __host__ void Create(cudaStream_t& renderStream);
        __host__ void Destroy();

        AssetHandle<Host::Vector<LineSegment>> m_hostLineSegments;
    };*/
};