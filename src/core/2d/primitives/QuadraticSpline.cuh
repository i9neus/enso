#pragma once

#include "Primitive2D.cuh"
#include "core/math/Polynomial.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }

    class QuadraticSpline
    {
    private:
        vec3 m_abc[2];

    public:
        __host__ __device__ QuadraticSpline() noexcept :
            m_abc{ vec3(0.0f), vec3(0.0f) } {}

        // Specify spline in terms of a native pair of quadratics
        __host__ __device__ QuadraticSpline(const vec3& v0, const vec3& v1) noexcept :
            m_abc{ v0, v1 } {}
        
        // Specific spline in terms of 3 control knots
        __host__ __device__ QuadraticSpline(const vec2& k0, const vec2& k1, const vec2& k2) noexcept
        {
            const mat3 M = transpose(mat3(vec3(2.0, -4.0, 2.0), vec3(-3.0, 4.0, -1.0), vec3(1.0, 0.0, 0.0)));
            m_abc[0] = M * vec3(k0.x, k1.x, k2.x);
            m_abc[1] = M * vec3(k0.y, k1.y, k2.y);
        }
        
        __host__ __device__ vec2                    PerpendicularPoint(const vec2& p) const;
        __host__ __device__ vec4                    EvaluateOverlay(const vec2& p, const OverlayCtx& ctx) const;
        __host__ __device__ bool                    TestPoint(const vec2& p, const float& thickness) const;
        __host__ __device__ bool                    IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const;
        __host__ __device__ bool                    Intersects(const BBox2f& bBox) const;
        __host__ __device__ BBox2f                  GetBoundingBox() const;
        
        __host__ __device__ __forceinline__ vec2 Evaluate(const float& t) const 
        { 
            return vec2(Poly::Quadratic::Evaluate(m_abc[0], t), Poly::Quadratic::Evaluate(m_abc[1], t));

        }
        __host__ __device__ __forceinline__ vec2 DEvaluate(const float& t) const
        {
            return vec2(Poly::Quadratic::DEvaluate(m_abc[0], t), Poly::Quadratic::DEvaluate(m_abc[1], t));
        }

        __host__ bool Serialise(Json::Node& rootNode, const int flags) const;
        __host__ bool Deserialise(const Json::Node& rootNode, const int flags);
    };

}