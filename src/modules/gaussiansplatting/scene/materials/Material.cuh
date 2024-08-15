#pragma once

#include "core/assets/GenericObject.cuh"
#include "core/3d/Ctx.cuh"

namespace Enso
{
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Material : public Device::GenericObject
        {
        public:
            __device__ virtual vec3                 Evaluate(const HitCtx& hit) const = 0;
        };
    }

    namespace Host
    {        
        class Material : public Host::GenericObject
        {
        public:
            __host__ Device::Material* GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            __host__ Material(const Asset::InitCtx& initCtx) : GenericObject(initCtx) {}
            __host__ void               SetDeviceInstance(Device::Material* deviceInstance) { cu_deviceInstance = deviceInstance; }
        protected:
            Device::Material*           cu_deviceInstance = nullptr;
        };
    }
}