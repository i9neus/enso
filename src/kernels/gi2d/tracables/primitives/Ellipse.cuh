#pragma once

#include "Primitive2D.cuh"

using namespace Cuda;

namespace Cuda
{
    namespace Host { template<typename T> class Vector; }
}

namespace GI2D
{
    class Ellipse : public Primitive2D
    {
    private:
        vec2 m_v[2];
        vec2 m_dv;

    public:
        __host__ __device__ Ellipse() noexcept : Primitive2D(), m_v{ vec2(0.0f), vec2(0.0f) }, m_dv(0.0f) {}
        __host__ __device__ Ellipse(const vec2& v0, const vec2& v1, const uchar flags, const vec3& col) noexcept :
            Primitive2D(flags, col), m_v{ v0, v1 }, m_dv(v1 - v0) {}

        __host__ __device__ virtual float                   Evaluate(const vec2& p, const float& dPdXY) const override final;

        __host__ __device__ __forceinline__ virtual BBox2f GetBoundingBox() const override final
        {
            return BBox2f(vec2(fminf(m_v[0].x, m_v[1].x), fminf(m_v[0].y, m_v[1].y)),
                vec2(fmaxf(m_v[0].x, m_v[1].x), fmaxf(m_v[0].y, m_v[1].y)));
        }

        __host__ __device__ void Set(const uint& idx, const vec2& v)
        {
            m_v[idx] = v;
            m_dv = m_v[1] - m_v[0];
        }

        __host__ __device__ __forceinline__ Ellipse& operator+=(const vec2& v)
        {
            m_v[0] += v; m_v[1] += v; m_dv = m_v[1] - m_v[0];
            return *this;
        }

        __host__ __device__ __forceinline__ Ellipse& operator-=(const vec2& v)
        {
            m_v[0] -= v; m_v[1] -= v; m_dv = m_v[1] - m_v[0];
            return *this;
        }

        __host__ __device__ __forceinline__ const vec2& operator[](const uint& idx) const { return m_v[idx]; }
        __host__ __device__ __forceinline__ vec2& dv(const uint& idx) { return m_dv; }
    };

    __host__ void GenerateRandomEllipses(Cuda::Host::Vector<Ellipse>& segments, const BBox2f& bounds, const ivec2 numSegmentsRange, const vec2 sizeRange, const uint seed);
};