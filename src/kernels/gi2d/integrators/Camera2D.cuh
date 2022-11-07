#pragma once

#include "kernels/CudaAsset.cuh"
#include "../Transform2D.cuh"
#include "../RenderCtx.cuh"

namespace GI2D
{
    namespace Device
    {
        class Camera2D : public Cuda::Device::Asset
        {
        public:
            __host__ __device__ Camera2D() {}

            __device__ virtual bool CreateRay(Ray2D& ray, RenderCtx& renderCtx) const = 0;

            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) = 0;
        };
    }
}      