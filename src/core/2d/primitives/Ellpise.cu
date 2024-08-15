#include "Ellipse.cuh"
#include "core/containers/Vector.cuh"
#include "core/math/Math.cuh"
#include "GenericIntersector.cuh"
#include "core/2d/sdf/SDF2DRenderer.cuh"
#include "core/math/ColourUtils.cuh"

#include <random>

namespace Enso
{
    __host__ __device__ vec4 Ellipse::EvaluateOverlay(const vec2& p, const OverlayCtx& ctx) const
    {
        vec4 L(0.f);
        if (ctx.HasFill())
        {
            L = Blend(L, ctx.fillColour * SDF2DRenderer::Ellipse(p, m_origin, m_radius, ctx.dPdXY));
        }
        if (ctx.HasStroke())
        {
            L = Blend(L, ctx.strokeColour * SDF2DRenderer::Torus(p, m_origin, m_radius, ctx.strokeThickness, ctx.dPdXY));
        }
        return L;
    }    

    __host__ __device__ bool Ellipse::Contains(const vec2& p, const float&) const
    {
        return length2(p - m_origin) < sqr(m_radius);
    }

    __host__ __device__ bool Ellipse::IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const
    {
        RayRange2D hitRange;
        if (!IntersectRayCircle(ray, m_origin, m_radius, hitRange)) { return false; }

        if (hitRange.tNear > 0.0 && hitRange.tNear < hit.tFar)
        {
            hit.n = (ray.o - m_origin + ray.d * hitRange.tNear) / m_radius;
            hit.tFar = hitRange.tNear;
        }
        else if (hitRange.tFar > 0.0 && hitRange.tFar < hit.tFar)
        {
            hit.n = (ray.o - m_origin + ray.d * hitRange.tFar) / m_radius;
            hit.tFar = hitRange.tFar;
        }
        else { return false; }

        return true;
    }
}