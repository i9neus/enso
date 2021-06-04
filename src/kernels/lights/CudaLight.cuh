#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"

namespace Cuda
{
    namespace Host { class Light; }

    namespace Device
    {
        class Light : public Device::Asset, public AssetTags<Host::Light, Device::Light>
        {
        public:
            Light() = default;

            __device__ inline void SetTransform(const BidirectionalTransform& transform) { m_transform = transform; }

        protected:
            __device__ Light(const BidirectionalTransform& transform) : m_transform(transform) {}
            __device__ ~Light() = default;

            BidirectionalTransform        m_transform;
        };
    }

    namespace Host
    {
        class Light : public Host::Asset, public AssetTags<Host::Light, Device::Light>
        {
        public:
            __host__ virtual Device::Light* GetDeviceInstance() const = 0;
        };
    }
}