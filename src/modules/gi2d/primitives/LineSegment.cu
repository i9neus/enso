#include "LineSegment.cuh"
#include "core/Vector.cuh"
#include "SDF.cuh"

#include <random>

namespace Enso
{    
    __host__ __device__ vec2 LineSegment::PerpendicularPoint(const vec2& p) const
    {
        return SDF::PerpLine(p, m_v[0], m_dv);
    }
    
    __host__ __device__ vec4 LineSegment::EvaluateOverlay(const vec2& p, const OverlayCtx& ctx) const
    {
        return SDF::Renderer::Line(p, m_v[0], m_dv, ctx.strokeThickness, ctx.dPdXY) * ctx.strokeColour;
    }

    __host__ __device__ bool LineSegment::TestPoint(const vec2& p, const float& thickness) const
    {
        return length2(p - SDF::PerpLine(p, m_v[0], m_dv)) < sqr(thickness);
    }

    __host__ __device__ bool LineSegment::IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const
    {
        // The intersection of the ray with the line
        vec2 n = vec2(m_dv.y, -m_dv.x);
        float tRay = (dot(n, m_v[0]) - dot(n, ray.o)) / dot(n, ray.d);

        if (tRay < 0.0f || tRay >= hit.tFar) { return false; }

        n = vec2(ray.d.y, -ray.d.x);
        float tSeg = (dot(n, ray.o) - dot(n, m_v[0])) / dot(n, m_dv);

        if (tSeg < 0.0f || tSeg > 1.0f) { return false; }
     
        n = vec2(m_dv.y, -m_dv.x);
        hit.n = n * (float(dot(n, ray.o - m_v[0]) > 0.0f) * 2.0f - 1.0f);
        hit.tFar = tRay;
        hit.kickoff = 1e-3f;

        return true;
    }

    __host__ __device__ bool LineSegment::Intersects(const BBox2f& bBox) const
    {
        if (bBox.Contains(m_v[0]) || bBox.Contains(m_v[1])) { return true; }

        // Ray-box intersection
        vec2 tNearPlane, tFarPlane;
        for (int dim = 0; dim < 2; dim++)
        {
            if (fabs(m_dv[dim]) > 1e-10f)
            {
                float t0 = (bBox.upper[dim] - m_v[0][dim]) / m_dv[dim];
                float t1 = (bBox.lower[dim] - m_v[0][dim]) / m_dv[dim];
                if (t0 < t1) { tNearPlane[dim] = t0;  tFarPlane[dim] = t1; }
                else { tNearPlane[dim] = t1;  tFarPlane[dim] = t0; }
            }
        }

        const float t0 = cwiseMax(tNearPlane);
        const float t1 = cwiseMin(tFarPlane);
        return t0 < t1&& t0 >= 0.f && t0 <= 1.f;
    }

    __host__ bool LineSegment::Serialise(Json::Node& rootNode, const int flags) const
    {
        return true;
    }

    __host__ bool LineSegment::Deserialise(const Json::Node& rootNode, const int flags)
    {
        return true;
    }

    __host__ void GenerateRandomLineSegments(Host::Vector<LineSegment>& segments, const BBox2f& bounds, const ivec2 numSegmentsRange, const vec2 sizeRange, const uint seed)
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

            segments[segIdx] = LineSegment(p + m_dv, p - m_dv);
        }
    }
}