#include "Ellipse.cuh"
#include "core/Vector.cuh"
#include "core/math/Math.cuh"
#include "GenericIntersector.cuh"
#include "SDF.cuh"

#include <random>

namespace Enso
{
    __host__ __device__ vec4 Ellipse::EvaluateOverlay(const vec2& p, const OverlayCtx& ctx) const
    {
        return SDF::Renderer::Torus(p, vec2(0.f), m_radius, ctx.strokeThickness, ctx.dPdXY) * ctx.strokeColour;
    }    

    __host__ __device__ bool Ellipse::Contains(const vec2& p, const float&) const
    {
        return length2(p) < sqr(m_radius);
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