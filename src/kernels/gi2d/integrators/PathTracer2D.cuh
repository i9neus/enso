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
            __host__ __device__ PathTracer2D(const SceneDescription& scene);

            //__device__ void Prepare(const SceneDescription* scene);
            __device__ void Integrate(RenderCtx& renderCtx, Camera2D& camera);

        private:
            const SceneDescription& m_scene;
        };
    }
}