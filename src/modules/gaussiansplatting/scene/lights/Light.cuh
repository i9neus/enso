#pragma once

#include "../tracables/Tracable.cuh"

namespace Enso
{
    struct LightParams
    {
        __device__ void Validate() const {}

        float radiance;
    };
    
    namespace Device
    {
        class Light : public Device::Tracable
        {
        public:
            __device__      Light() {}
            __device__      ~Light() noexcept {}
            __device__      void Synchronise(const LightParams& params) { m_params = params; }
            __device__      void Verify() {}

            __device__ virtual float Sample(const Ray& incident, Ray& extant, const HitCtx& hit, const vec2& xi) = 0;
            __device__ virtual float Evaluate(Ray& extant, const HitCtx& hit) = 0;


        protected:
            LightParams     m_params;
        };
    }

    namespace Host
    {
        class Light : public Host::Tracable
        {
        public:
            __host__ virtual            ~Light() noexcept {}
            __host__ Device::Light*     GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            __host__ Light(const Asset::InitCtx& initCtx) :
                Tracable(initCtx)
            {}

            __host__ void SetDeviceInstance(Device::Light* deviceInstance)
            {
                Tracable::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Tracable>(deviceInstance));
                cu_deviceInstance = deviceInstance;
            }

            __host__ virtual void OnSynchroniseTracable(const uint syncFlags) override final
            {
                if (syncFlags == kSyncParams)
                {
                    SynchroniseObjects<Device::Light>(cu_deviceInstance, m_params);
                }
                OnSynchroniseLight(syncFlags);
            }

            __host__ virtual void OnSynchroniseLight(const uint syncFlags) = 0;


        private:
            Device::Light*              cu_deviceInstance = nullptr;
            LightParams                 m_params;
        };
    }
}