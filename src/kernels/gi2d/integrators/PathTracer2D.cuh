#pragma once

#include "../BIH2DAsset.cuh"
#include "../SceneDescription.cuh"
#include "Camera2D.cuh"

using namespace Cuda;

namespace GI2D
{
    namespace Device
    {        
        class PathTracer2D : public Cuda::Device::Asset
        {
            enum _attrs : int { kInvalidHit = -1 };

        public:
            __device__ PathTracer2D(const Device::SceneDescription& scene) : m_scene(scene) {}

            __device__ void Integrate(RenderCtx& renderCtx) const;

        private:
            __device__ int Trace(const Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx, const Core::Vector<Device::Tracable*>& tracables) const;

            const Device::SceneDescription& m_scene;
        };
    }
}