#include "Ellipse.cuh"
#include "core/Vector.cuh"
#include "core/math/Math.cuh"
#include "GenericIntersector.cuh"
#include "SDF.cuh"

#include <random>

namespace Enso
{
    __host__ __device__ float Ellipse::EvaluateOverlay(const vec2& p, const float& dPdXY) const
    {
        return SDF::Renderer::Circle(p, vec2(0.f), m_radius, 6.f, dPdXY);
    }    

    __host__ __device__ bool Ellipse::Contains(const vec2& p, const float&) const
    {
        return length2(p) < sqr(m_radius);
    }

    __host__ __device__ bool Ellipse::IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const
    {
        // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
        /*vec2 o = ray.o - m_origin;
        float a = dot(ray.d, ray.d);
        float b = 2.0 * dot(ray.d, o);
        float c = dot(o, o) - sqr(m_radius);

        float b2ac4 = b * b - 4.0 * a * c;
        if (b2ac4 < 0.0) { return false; }

        float sqrtb2ac4 = sqrt(b2ac4);
        float t0 = (-b + sqrtb2ac4) / (2.0 * a);
        float t1 = (-b - sqrtb2ac4) / (2.0 * a);

        if (t1 < t0) { swap(t0, t1); }

        if (t0 > 0.0 && t0 < hit.tFar)
        {
            hit.n = (o + ray.d * t0) / m_radius;
            hit.tFar = t0;
        }
        else if (t1 > 0.0 && t1 < hit.tFar)
        {
            hit.n = (o + ray.d * t1) / m_radius;
            hit.tFar = t1;
        }
        else { return false; }*/

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