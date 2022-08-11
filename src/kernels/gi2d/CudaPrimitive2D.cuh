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
    
    enum GI2DLineSegmentFlags : int
    {
        kSelected = 1
    };

    class LineSegment
    {
    public:
        __host__ __device__ LineSegment() noexcept : flags(0) {}
        __host__ __device__ LineSegment(const vec2& v0, const vec2& v1) noexcept :
            flags(0), v(v0), dv(v1 - v0) {}

        __host__ __device__ __forceinline__ vec2 PerpendicularPoint(const vec2& p) const;
        __host__ __device__ float Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const;
        __host__ __device__ bool TestPoint(const vec2& p, const float& thickness) const;
        __host__ __device__ float TestRay(const vec2& o, const vec2& d) const;
        __host__ __device__ vec2 PointAt(const float& t) const { return v + dv * t; }

        __host__ __device__ __forceinline__ BBox2f GetBoundingBox() const
        {
            return BBox2f(vec2(min(v.x, v.x + dv.x), min(v.y, v.y + dv.y)),
                vec2(max(v.x, v.x + dv.x), max(v.y, v.y + dv.y)));
        }

        uchar flags;
        vec2 v, dv;
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