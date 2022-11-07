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
        public:
            __device__ PathTracer2D(const Device::SceneDescription& scene) : m_scene(scene) {}

            __device__ void Integrate(RenderCtx& renderCtx);

        private:
            const Device::SceneDescription& m_scene;
        };
    }
}