#include "PathTracer2D.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"

#include "../RenderCtx.cuh"

using namespace Cuda;

namespace GI2D
{
    __device__ Device::PathTracer2D::PathTracer2D(const SceneDescription& scene) : 
        m_scene(scene)
    {}

    __device__ void Device::PathTracer2D::Integrate(RenderCtx& renderCtx, Camera2D& camera)
    {    
        assert(m_scene.bih && m_scene.tracables);
        
        Ray2D ray;
        if (!camera.CreateRay(ray, renderCtx)) { return; }

        const auto& tracables = *m_scene.tracables;
        const auto& bih = *m_scene.bih;
        
        for (int depth = 0; depth < 1; ++depth)
        {
            float theta = renderCtx.rng.Rand<0>() * kTwoPi;
            HitCtx2D hit;
            int hitIdx = 0;

            auto onIntersect = [&, this](const uint* primRange, RayRange2D& range) -> float
            {
                for (uint idx = primRange[0]; idx < primRange[1]; ++idx)
                {
                    if (tracables[idx]->IntersectRay(ray, hit))
                    {
                        if (hit.tFar < range.tFar)
                        {
                            range.tFar = hit.tFar;
                            hitIdx = idx;
                        }
                    }
                }
            };
            bih.TestRay(ray, kFltMax, onIntersect);

            if (hit.tFar != kFltMax)
            {
                renderCtx.accumulator.Accumulate(vec4(tracables[hitIdx]->GetColour(), 0.0f), renderCtx);
            }
        }

        renderCtx.accumulator.Accumulate(vec4(kZero, 1.0f), renderCtx);
    }
    DEFINE_KERNEL_PASSTHROUGH(Integrate);   
}