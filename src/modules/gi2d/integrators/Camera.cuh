#pragma once

#include "../FwdDecl.cuh"
#include "core/AssetAllocator.cuh"
#include "PathTracer2D.cuh"

namespace Enso
{
    class Ray2D;
    class HitCtx2D;
    class RenderCtx;
    class UIViewCtx;
    
    namespace Device
    {   
        class Camera 
        {
        public:
            __device__ virtual void Prepare(const uint dirtyFlags);
            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const = 0;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) = 0;
            __device__ void Synchronise(const Device::SceneDescription& scene);

        protected:
            __device__ Camera();

            __device__ virtual void Integrate(const uchar flags);

        private:
            Device::PathTracer2D                    m_voxelTracer;
            int                                     m_frameIdx;
        };
    }

    namespace Host
    {
        class Camera
        {
        public:
            __host__ virtual void Render() = 0;
            __host__ virtual bool Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx) = 0;
            __host__ virtual Device::Camera* GetDeviceInstance() const = 0;

        protected:
            Camera() = default;
        };
    };
}      