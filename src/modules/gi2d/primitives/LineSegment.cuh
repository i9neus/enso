#pragma once

#include "Primitive2D.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }

    class LineSegment : public Primitive2D
    {
    private:
        vec2 m_v[2];
        vec2 m_dv;
    public:
        __host__ __device__ LineSegment() noexcept : Primitive2D(), m_v{ vec2(0.0f), vec2(0.0f) }, m_dv(0.0f) {}
        __host__ __device__ LineSegment(const vec2& v0, const vec2& v1, const uchar flags, const vec3& col) noexcept :
            Primitive2D(flags, col), m_v{ v0, v1 }, m_dv(v1 - v0) {}
        
        __host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const override final;
        __host__ __device__ virtual float                   EvaluateOverlay(const vec2& p, const float& dPdXY) const override final;
        __host__ __device__ virtual bool                    TestPoint(const vec2& p, const float& thickness) const override final;
        __host__ __device__ virtual bool                    IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const override final;
        __host__ __device__ virtual bool                    Intersects(const BBox2f& bBox) const override final;

        __host__ __device__ vec2                            PointAt(const float& t) const { return m_v[0] + m_dv * t; }

        __host__ __device__ __forceinline__ virtual BBox2f GetBoundingBox() const override final
        {
            return LineBBox2(m_v[0], m_v[1]);
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

        __host__ __device__ __forceinline__ vec2& operator[](const uint& idx) { return m_v[idx]; }
        __host__ __device__ __forceinline__ const vec2& operator[](const uint& idx) const { return m_v[idx]; }
        __host__ __device__ __forceinline__ vec2& dv(const uint& idx) { return m_dv; }

        __host__ bool Serialise(Json::Node& rootNode, const int flags) const;
        __host__ bool Deserialise(const Json::Node& rootNode, const int flags);
    };

    __host__ void GenerateRandomLineSegments(Host::Vector<LineSegment>& segments, const BBox2f& bounds, const ivec2 numSegmentsRange, const vec2 sizeRange, const uint seed);
}