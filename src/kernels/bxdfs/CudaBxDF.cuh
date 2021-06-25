﻿#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"

namespace Cuda
{
    namespace Host { class BxDF; }

    namespace Device
    {
        class BxDF : public Device::RenderObject, public AssetTags<Host::BxDF, Device::BxDF>
        {
        public:
            BxDF() = default;

        protected:
            __device__ ~BxDF() = default;
        };
    }

    namespace Host
    {
        class BxDF : public Host::RenderObject, public AssetTags<Host::Tracable, Device::Tracable>
        {
        public:
            __host__ virtual Device::BxDF* GetDeviceInstance() const = 0;
        };
    }
}