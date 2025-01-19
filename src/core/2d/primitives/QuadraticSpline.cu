#include "QuadraticSpline.cuh"
#include "core/containers/Vector.cuh"
#include "core/2d/sdf/SDF2DRenderer.cuh"

#include <random>

namespace Enso
{    
    __host__ __device__ vec2 QuadraticSpline::PerpendicularPoint(const vec2& p) const
    {
        float tPerp;
        vec2 xyPerp;
        return (SDF2D::QuadradicSplinePerpPointApprox(p, m_abc[0], m_abc[1], 0., tPerp, xyPerp)) ? xyPerp : vec2(0.);
    }

    __host__ __device__ vec4 QuadraticSpline::EvaluateOverlay(const vec2& p, const OverlayCtx& ctx) const
    {
        const float stroke = SDF2DRenderer::QuadraticSpline(p, m_abc[0], m_abc[1], ctx.strokeThickness, ctx.dPdXY);
        return vec4(ctx.strokeColour.xyz, ctx.strokeColour.w * stroke);
    }

    __host__ __device__ bool QuadraticSpline::TestPoint(const vec2& p, const float& thickness) const
    {
        return length2(p - PerpendicularPoint(p)) < sqr(thickness);
    }

    __host__ __device__ bool QuadraticSpline::IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const
    {
        return false;
    }

    __host__ __device__ bool QuadraticSpline::Intersects(const BBox2f& bBox) const
    {
        return false;
    }

    __host__ __device__ BBox2f QuadraticSpline::GetBoundingBox() const
    {
        BBox2f bBox = BBox2f::Invalid();
        for (int dim = 0; dim < 2; ++dim)
        {
            // Update the bounds based on the curve's start and endpoints
            float p = Poly::Quadratic::Evaluate(m_abc[dim], 0);
            bBox[0][dim] = fminf(bBox[0][dim], p);
            bBox[1][dim] = fmaxf(bBox[1][dim], p);
            p = Poly::Quadratic::Evaluate(m_abc[dim], 1);
            bBox[0][dim] = fminf(bBox[0][dim], p);
            bBox[1][dim] = fmaxf(bBox[1][dim], p);

            // If the polynomaial has a critical point inside the interval (0, 1), do a bounds check
            const float t = Poly::Quadratic::CriticalPoint(m_abc[dim]);
            if (t > 0 && t < 1)
            {
                p = Poly::Quadratic::Evaluate(m_abc[dim], t);
                bBox[0][dim] = fminf(bBox[0][dim], p);
                bBox[1][dim] = fmaxf(bBox[1][dim], p);
            }
        } 

        return bBox;
    }

    __host__ bool QuadraticSpline::Serialise(Json::Node& rootNode, const int flags) const
    {
        return true;
    }

    __host__ bool QuadraticSpline::Deserialise(const Json::Node& rootNode, const int flags)
    {
        return true;
    }
}