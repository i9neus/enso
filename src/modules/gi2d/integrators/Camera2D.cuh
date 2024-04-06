#pragma once

#include "../Dirtiness.cuh"

namespace Enso
{
    class Ray2D;
    class HitCtx2D;
    class RenderCtx;
    
    namespace Device
    {
        class ICamera2D
        {
        public:
            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const = 0;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) = 0;

        protected:
            __host__ __device__ ICamera2D() {}
        };
    }

    namespace Host
    {
        class ICamera2D
        {
        public:
            ICamera2D() = delete;
        };
    };
}      