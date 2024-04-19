#pragma once

#include "../bih/BIH2DAsset.cuh"
#include "../scene/SceneContainer.cuh"
#include "../RenderCtx.cuh"

namespace Enso
{
    namespace Device
    {
        class PathTracer2D : public Device::Asset
        {
            enum _attrs : int { kInvalidHit = -1 };

        public:
            __device__ PathTracer2D() {}

            __device__ void Synchronise(const Device::SceneContainer& scene) 
            { 
                scene.Validate();  
                m_scene = scene; 
            }

            __device__ void Integrate(RenderCtx& renderCtx) const;

        private:
            __device__ bool SelectLight(const Ray2D& incident, const HitCtx2D& hitCtx, const float& xi, int& lightIdx, float& weight) const;

            __device__ int Trace(const Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const;
            __device__ bool Shade(Ray2D& ray, const HitCtx2D& hit, RenderCtx& renderCtx) const;

            Device::SceneContainer m_scene;
        };
    }
}