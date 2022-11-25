#pragma once

#include "../bih/BIH2DAsset.cuh"
#include "../SceneDescription.cuh"
#include "Camera2D.cuh"

namespace Enso
{
    namespace Device
    {
        class PathTracer2D : public Device::Asset
        {
            enum _attrs : int { kInvalidHit = -1 };

        public:
            __device__ PathTracer2D(const Device::SceneDescription& scene) : m_scene(scene) {}

            __device__ void Integrate(RenderCtx& renderCtx) const;

        private:
            __device__ int Trace(const Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const;
            __device__ bool Shade(Ray2D& ray, const HitCtx2D& hit, RenderCtx& renderCtx) const;

            const Device::SceneDescription& m_scene;
        };
    }
}