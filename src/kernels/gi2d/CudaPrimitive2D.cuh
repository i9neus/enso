#pragma once

#include "generic/StdIncludes.h"
#include "kernels/math/bbox/CudaBBox2.cuh"
#include <map>

#include "../CudaAsset.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"

namespace Cuda
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
        __host__ __device__ Primitive2D(const uchar& f) noexcept : m_flags(f) {}

        uchar m_flags;

    public:
        __host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const = 0;
        __host__ __device__ virtual float                   Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const = 0;
        __host__ __device__ virtual bool                    TestPoint(const vec2& p, const float& thickness) const = 0;
        __host__ __device__ virtual float                   TestRay(const vec2& o, const vec2& d) const = 0;
        __host__ __device__ virtual BBox2f                  GetBoundingBox() const = 0;

        __host__ __device__ __forceinline__ bool            IsSelected() const { return m_flags & k2DPrimitiveSelected; }
        __host__ __device__ __forceinline__ void            SetFlags(const uchar flags) { m_flags |= flags; }
        __host__ __device__ __forceinline__ void            UnsetFlags(const uchar flags) { m_flags &= ~flags; }
    };
    
    class LineSegment : public Primitive2D
    {
    private:
        vec2 m_v[2];
        vec2 m_dv;
    public:
        __host__ __device__ LineSegment() noexcept : Primitive2D(), m_v{ vec2(0.0f), vec2(0.0f) }, m_dv(0.0f) {}
        __host__ __device__ LineSegment(const vec2& v0, const vec2& v1, const uchar flags) noexcept :
            Primitive2D(flags), m_v{ v0, v1 }, m_dv(v1 - v0) {}

        __host__ __device__  virtual vec2                   PerpendicularPoint(const vec2& p) const override final;
        __host__ __device__ virtual float                   Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const override final;
        __host__ __device__ virtual bool                    TestPoint(const vec2& p, const float& thickness) const override final;
        __host__ __device__ virtual float                   TestRay(const vec2& o, const vec2& d) const override final;
        
        __host__ __device__ __forceinline__ virtual BBox2f GetBoundingBox() const override final
        {
            return BBox2f(vec2(min(m_v[0].x, m_v[1].x), min(m_v[1].y, m_v[1].y)),
                          vec2(max(m_v[0].x, m_v[1].x), max(m_v[1].y, m_v[1].y)));
        }

        __host__ __device__ vec2                            PointAt(const float& t) const { return m_v[0] + m_dv * t; }

        __host__ __device__ __forceinline__ vec2& operator[](const uint& idx) { return m_v[idx]; }
        __host__ __device__ __forceinline__ vec2& dv(const uint& idx) { return m_dv; }
    };

    __host__ void GenerateRandomLineSegments(Host::Vector<LineSegment>& segments, const BBox2f& bounds, const ivec2 numSegmentsRange, const vec2 sizeRange, const uint seed);
    
    /*class Primitive2DContainer
    {
    public:
        __host__ Primitive2DContainer();
        
        __host__ void Create(cudaStream_t& renderStream);
        __host__ void Destroy();

        AssetHandle<Host::Vector<LineSegment>> m_hostLineSegments;
    };*/
};