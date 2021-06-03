#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"

namespace Cuda
{
    namespace Host 
    {  
        class Tracable;
        class Sphere;
    }
    
    namespace Device
    {
        class Tracable : public Device::Asset, public AssetTags<Host::Tracable, Device::Tracable>
        {
        public:
            Tracable() = default;

            __device__ inline void SetTransform(const BidirectionalTransform& transform) { m_transform = transform; }

        protected:
            __device__ Tracable(const BidirectionalTransform& transform) : m_transform(transform) {}
            __device__ ~Tracable() = default;

            BidirectionalTransform        m_transform;
        };
    }

    namespace Host
    {
        class Tracable : public Host::Asset, public AssetTags<Host::Tracable, Device::Tracable>
        {
        public:
            __host__ virtual Device::Tracable* GetDeviceInstance() const = 0;
        };
    }
}