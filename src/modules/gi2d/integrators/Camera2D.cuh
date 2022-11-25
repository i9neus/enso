#pragma once

#include "core/Asset.cuh"
#include "../Transform2D.cuh"
#include "../RenderCtx.cuh"

namespace Enso
{
    namespace Device
    {
        class Camera2D : public Device::Asset
        {
        public:
            __host__ __device__ Camera2D() {}

            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const = 0;

            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) = 0;
        };
    }
}      