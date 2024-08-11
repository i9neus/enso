#pragma once

#include "Material.cuh"

namespace Enso
{
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Lambertian : public Device::Material
        {
        public:
            __device__ virtual vec3 Sample(const HitCtx& hit) const override final
            {

            }

            __device__ virtual vec3 Evaluate(const HitCtx& hit) const override final
            {
            }
        };
    }

    namespace Host
    {
        class Lambertian : public Host::Material
        {
        public:
            __host__ Device::Lambertian* GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            __host__ Lambertian(const Asset::InitCtx& initCtx) : Material(initCtx) {}
            __host__ void               SetDeviceInstance(Device::Lambertian* deviceInstance) { cu_deviceInstance = deviceInstance; }
        protected:
            Device::Lambertian* cu_deviceInstance = nullptr;
        };
    }
}