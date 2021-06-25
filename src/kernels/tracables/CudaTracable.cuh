#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "CudaGenericIntersectors.cuh"
#include "../CudaRenderObject.cuh"

namespace Cuda
{
    namespace Host  { class Tracable;  }
    
    namespace Device
    {
        class Tracable : public Device::RenderObject, public AssetTags<Host::Tracable, Device::Tracable>
        {
        public:
            Tracable() = default;

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const = 0;
            __device__ virtual void InitialiseKernelConstantData() const {};

        protected:
            __device__ Tracable() = default;
            __device__ virtual ~Tracable() = default;
        };
    }

    namespace Host
    {
        class Tracable : public Host::RenderObject, public AssetTags<Host::Tracable, Device::Tracable>
        {
        protected:
            __host__ Tracable() = default;
            __host__ virtual ~Tracable() = default;

        public:
            __host__ virtual Device::Tracable* GetDeviceInstance() const = 0;
        };
    }
}