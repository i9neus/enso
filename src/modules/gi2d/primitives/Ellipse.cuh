#pragma once

#include "Primitive2D.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }

    class Ellipse
    {
    private:
        vec2    m_origin;
        float   m_radius;

    public:
        __host__ __device__ Ellipse() : m_origin(0.f), m_radius(0.f) {}
        __host__ __device__ Ellipse(const vec2& o, const float& r) :
            m_origin(o),
            m_radius(r) {}

        __host__ __device__ vec4                    EvaluateOverlay(const vec2& p, const OverlayCtx & ctx) const;
        __host__ __device__ bool                    Contains(const vec2& p, const float& dPdXY) const;        
        __host__ __device__ bool                    IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const;

        __host__ void                               SetOrigin(const vec2& origin) { m_origin = origin; }
        __host__ void                               SetRadius(const float& radius) { m_radius = radius; }

        __host__ __device__ __forceinline__ BBox2f  GetBoundingBox() const
        {
            return CircleBBox2(m_origin, m_radius);
        }
    };

    __host__ void GenerateRandomEllipses(Host::Vector<Ellipse>& segments, const BBox2f& bounds, const ivec2 numSegmentsRange, const vec2 sizeRange, const uint seed);
}