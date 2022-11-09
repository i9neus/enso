#include "PathTracer2D.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"

#include "../RenderCtx.cuh"

using namespace Cuda;

namespace GI2D
{
    __device__ int Device::PathTracer2D::Trace(const Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx, const Core::Vector<Device::Tracable*>& tracables) const
    {
        hit.tracableIdx = kInvalidHit;

        const auto& bih = *m_scene.tracableBIH;
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

        return hit.tracableIdx;
    }
    
    __device__ void Device::PathTracer2D::Integrate(RenderCtx& renderCtx) const
    {    
        assert(m_scene.tracableBIH && m_scene.tracables);
        
        Ray2D ray;
        if (!renderCtx.camera.CreateRay(ray, renderCtx)) { return; }

        const auto& tracables = *m_scene.tracables;
        HitCtx2D hit;
        for (int depth = 0; depth < 1; ++depth)
        {
            if (Trace(ray, hit, renderCtx, tracables) == kInvalidHit) { break; }

            if (hit.tFar != kFltMax)
            {
                renderCtx.camera.Accumulate(vec4(tracables[hit.tracableIdx]->GetColour(), 0.0f), renderCtx);
            }          
        }

        renderCtx.camera.Accumulate(vec4(kZero, 1.0f), renderCtx);
    }
    DEFINE_KERNEL_PASSTHROUGH(Integrate);   
}