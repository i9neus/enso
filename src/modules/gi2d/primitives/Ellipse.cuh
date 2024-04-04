#pragma once

#include "Primitive2D.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }

    class Ellipse : public Primitive2D
    {
    private:
        vec2    m_origin;
        float   m_radius;

    public:
        __host__ __device__ Ellipse() : Primitive2D() {}
        __host__ __device__ Ellipse(const vec2& o, const float& r) :
            Primitive2D(0, kOne),
            m_origin(o),
            m_radius(r) {}

        __host__ __device__ virtual float                   EvaluateOverlay(const vec2& p, const float& dPdXY) const override final;
        __host__ __device__ virtual bool                    Contains(const vec2& p, const float& dPdXY) const override final;

        __host__ __device__ virtual bool                    IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const override final;

        __host__ __device__ __forceinline__ virtual BBox2f  GetBoundingBox() const override final
        {
            return CircleBBox2(m_origin, m_radius);
        }
    };

    __host__ void GenerateRandomEllipses(Host::Vector<Ellipse>& segments, const BBox2f& bounds, const ivec2 numSegmentsRange, const vec2 sizeRange, const uint seed);
}