#include "PathTracer2D.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/Hash.h"

#include "../lights/Light.cuh"
#include "../RenderCtx.cuh"

namespace Enso
{
    __device__ int Device::PathTracer2D::Trace(const Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const
    {
        hit.tracableIdx = kInvalidHit;
        hit.flags = 0;

        const auto& bih = *m_scene.tracableBIH;
        const auto& tracables = *m_scene.tracables;
        auto onIntersect = [&tracables, &ray, &hit](const uint* primRange, RayRange2D& range) -> float
        {
            for (uint idx = primRange[0]; idx < primRange[1]; ++idx)
            {
                if (tracables[idx]->IntersectRay(ray, hit))
                {
                    if (hit.tFar < range.tFar)
                    {
                        range.tFar = hit.tFar;
                        hit.tracableIdx = idx;
                    }
                }
            }
        };
        bih.TestRay(ray, kFltMax, onIntersect);

        if (hit.tracableIdx != kInvalidHit)
        {
            hit.p = ray.PointAt(hit.tFar);
            hit.n = normalize(hit.n);
            hit.depth++;

            //if (renderCtx.IsDebug()) printf("Hit: %i: %f, [%f, %f] (%f, %f)\n", hit.tracableIdx, hit.tFar, hit.p.x, hit.p.y, hit.n.x, hit.n.y);
        }
        return hit.tracableIdx;
    }

    template<typename T>
    __device__ __inline__ __forceinline__ int LowerBound(int i0, int i1, const T* pmf, const float& xi)
    {
        while (i1 - i0 > 1)
        {
            const int iMid = i0 + (i1 - i0) / 2;
            if (pmf[iMid] < xi) { i0 = iMid; }
            else { i1 = iMid; }
        }

        if (pmf[i1] < xi) { return i1 + 1; }
        else if (pmf[i0] < xi) { return i0 + 1; }
        return i0;
    }

    __device__ bool Device::PathTracer2D::SelectLight(const Ray2D& incident, const HitCtx2D& hitCtx, const float& xi, int& lightIdx, float& weight) const
    {
        constexpr int kMaxLights = 5;
        float pmf[kMaxLights + 1];
        pmf[0] = 0.0f;
        const auto& lights = *(m_scene.lights);
        for (int idx = 0; idx < lights.Size() && idx < kMaxLights; ++idx)
        {
            float estimate = lights[idx]->Estimate(incident, hitCtx);

            pmf[1 + idx] = pmf[idx] + estimate;
        }

        // No lights are in range to sample
        if (pmf[lights.Size()] == 0.0f) { return false; }

        lightIdx = LowerBound(1, lights.Size(), pmf, xi * pmf[lights.Size()]) - 1;
        weight = pmf[lights.Size()] / (pmf[lightIdx + 1] - pmf[lightIdx]);

        assert(lightIdx >= 0 && lightIdx < lights.Size());
        return true;
    }

    __device__ bool Device::PathTracer2D::Shade(Ray2D& ray, const HitCtx2D& hit, RenderCtx& renderCtx) const
    {
        const vec4 xi = renderCtx.rng.Rand<0, 1, 2, 3>();

        // Sample the direct component
        if (xi.z < 0.5)
        {
            const auto& lights = *m_scene.lights;
            if (lights.IsEmpty()) { return false; }

            // Select a light to sample or evaluate
            int lightIdx = 0;
            float weightLight = 1.0f;
            if (lights.Size() > 1)
            {
                // Weight each light equally. Cheap to sample but noisy for scenes with lots of lights.
                //if (m_activeParams.lightSelectionMode == kLightSelectionNaive)
                if(false)
                {
                    lightIdx = max<int>(lights.Size() - 1, int(xi.x * lights.Size()));
                    weightLight = float(lights.Size());
                }
                // Build a PMF based on a crude estiamte of the irradiance at the shading point. Expensive, but 
                // significantly reduces noise. 
                else if (!SelectLight(ray, hit, xi.w, lightIdx, weightLight))
                {
                    return false;
                }
            }            

            // Sample the light
            vec2 extantLight;
            vec3 LLight;
            float pdfLight;
            if (!lights[lightIdx]->Sample(ray, hit, xi.y, extantLight, LLight, pdfLight)) { return false; }

            // Derive the extant ray from the incident ray
            ray.DeriveDirectSample(hit, extantLight, LLight * weightLight, lightIdx);
        }
        // Indirect light sampling	
        else
        {
            // Sample a 2D Lambertian BRDF
            vec2 d = vec2(xi.y, sqrt(1.0 - sqr(xi.y)));
            d = normalize(hit.n * d.y + vec2(hit.n.y, -hit.n.x) * d.x);

            //if (renderCtx.IsDebug()) printf("Scatter: [%f, %f] %f\n", hit.p.x, hit.p.y, hit.kickoff);

            ray.DeriveIndirectSample(hit, d, kOne);
        }

        return true;
    }

    __device__ void Device::PathTracer2D::Integrate(RenderCtx& renderCtx) const
    {
        assert(m_scene.tracableBIH && m_scene.tracables);

        Ray2D ray;
        HitCtx2D hit;
        if (!renderCtx.camera.CreateRay(ray, hit, renderCtx)) { return; }

        hit.debugData = renderCtx.debugData;
        //vec3 debug(kZero);
        //hit.debug = &debug;

        constexpr int kMaxDepth = 2;
        for (int depth = 0; depth < kMaxDepth; ++depth)
        {
            //if (renderCtx.IsDebug()) printf("%i: %f, %f -> %f, %f\n", depth, ray.o.x, ray.o.y, ray.d.x, ray.d.y);
            // Trace the rays generated by the shading pass
            if (Trace(ray, hit, renderCtx) == kInvalidHit) { break; }

            auto& tracable = *(*m_scene.tracables)[hit.tracableIdx];
            auto hitLightIdx = tracable.GetLightIdx();
            if (ray.IsDirectSample())
            {
                if (ray.lightIdx == hitLightIdx)
                {
                    renderCtx.camera.Accumulate(vec4(ray.throughput, 0.0f), renderCtx);
                }
                continue;
            }

            if (depth == kMaxDepth - 1 || hitLightIdx != kTracableNotALight) { break; }

            if (!Shade(ray, hit, renderCtx)) { break; }

            hit.PrepareNext();
        }

        //renderCtx.camera.Accumulate(vec4(hit.GetDebug(), 0.0f), renderCtx);
    }
    DEFINE_KERNEL_PASSTHROUGH(Integrate);
}
