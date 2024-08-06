#pragma once

#include "core/GenericObject.cuh"
#include "core/3d/Ctx.cuh"

namespace Enso
{
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Texture2D : public Device::GenericObject
        {
        public:
            __device__ virtual vec3                 Evaluate(const HitCtx& hit) const = 0;
        };
    }

    namespace Host
    {
        class Texture2D : public Host::GenericObject
        {
        public:
            __host__ Device::Texture2D* GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            __host__ Texture2D(const Asset::InitCtx& initCtx) : GenericObject(initCtx) {}
            __host__ void               SetDeviceInstance(Device::Texture2D* deviceInstance) { cu_deviceInstance = deviceInstance; }

        protected:
            Device::Texture2D* cu_deviceInstance = nullptr;
        };
    }
}