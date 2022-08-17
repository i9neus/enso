#include "CudaPrimitive2D.cuh"
#include "../CudaVector.cuh"

#include <random>

using namespace Cuda;

namespace GI2D
{
    __host__ __device__ ViewTransform::ViewTransform()
    {
        matrix = mat3::Indentity();
        scale = 1.f;
        rotate = 0.f;
        trans = vec2(0.f);
        mouseView = 0.0f;
        dPdXY = 1.0 / 1024.f;
    }
    
    __host__ __device__ vec2 LineSegment::PerpendicularPoint(const vec2& p) const
    {
       return m_v[0] + Cuda::saturate((dot(p, m_dv) - dot(m_v[0], m_dv)) / dot(m_dv, m_dv)) * m_dv;
    }

    __host__ __device__ float LineSegment::Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const
    {
        return Cuda::saturate(1.0f - (length(p - PerpendicularPoint(p)) - thickness) / dPdXY);
    }

    __host__ __device__ bool LineSegment::TestPoint(const vec2& p, const float& thickness) const
    {
        return length2(p - PerpendicularPoint(p)) < sqr(thickness);
    }

    __host__ __device__ float LineSegment::TestRay(const vec2& o, const vec2& d) const
    {
        // The intersection of the ray with the line
        vec2 n = vec2(m_dv.y, -m_dv.x);
        float tSeg = (dot(n, m_v[0]) - dot(n, o)) / dot(n, d);
         
        if (tSeg < 0.0f) { return kFltMax; }

        n = vec2(d.y, -d.x);
        float tRay = (dot(n, o) - dot(n, m_v[0])) / dot(n, m_dv);

        return (tRay >= 0.0 && tRay <= 1.0) ? tSeg : kFltMax;

        // Check to see whether it's bounded
        //return (length2((o + d * t - (v + m_dv * 0.5)) / (m_dv * 0.5)) > 1.0f) ? kFltMax : t;
    }

    __host__ void GenerateRandomLineSegments(Cuda::Host::Vector<LineSegment>& segments, const BBox2f& bounds, const ivec2 numSegmentsRange, const vec2 sizeRange, const uint seed)
    {
        std::mt19937 mt(seed);
        std::uniform_real_distribution<> realRng;
        std::uniform_int_distribution<> intRng;
        
        const int numSegments = numSegmentsRange[0] + intRng(mt) % max(1, numSegmentsRange[1] - numSegmentsRange[0]);
        segments.Resize(numSegments);
        for (int segIdx = 0; segIdx < numSegments; ++segIdx)
        {
            const vec2 p(mix(bounds.lower.x, bounds.upper.x, realRng(mt)), mix(bounds.lower.y, bounds.upper.y, realRng(mt)));
            const float theta = realRng(mt) * kPi;
            const float size = 0.5f * mix(sizeRange[0], sizeRange[1], std::pow(realRng(mt), 2.0f));
            const vec2 m_dv = vec2(std::cos(theta), std::sin(theta)) * size;

            segments[segIdx] = LineSegment(p + m_dv, p - m_dv, 0);
        }
    }
}