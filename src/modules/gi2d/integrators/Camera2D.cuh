#pragma once

#include "core/AssetAllocator.cuh"
#include "../SceneDescription.cuh"

namespace Enso
{
    class Ray2D;
    class HitCtx2D;
    class RenderCtx;
    class UIViewCtx;
    
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
            __host__ virtual void Render() = 0;
            __host__ virtual bool Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx) = 0;
            __host__ virtual Device::ICamera2D* GetDeviceInstance() const = 0;

        protected:
            ICamera2D() = default;
        };
    };
}      