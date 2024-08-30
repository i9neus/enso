#pragma once

#include "LightSampler.cuh"

namespace Enso
{
    namespace Device
    {
        class QuadLight : public Device::LightSampler
        {
        public:
            __device__      QuadLight() {}

            __device__ virtual float        Sample(const Ray& incident, Ray& extant, const HitCtx& hit, const vec2& xi) const override final;
            __device__ virtual float        Evaluate(Ray& extant, const HitCtx& hit) const override final;
        };
    }

    namespace Host
    {
        class QuadLight : public Host::LightSampler
        {
        public:
            __host__ QuadLight(const Asset::InitCtx& initCtx, const vec3& radiance, AssetHandle<Host::Tracable>& tracable);
            __host__ virtual ~QuadLight() noexcept;

        protected:
            __host__ virtual bool TryBind(AssetHandle<Host::Tracable>& tracable) override final;

            //__host__ virtual void OnSynchroniseLightSampler(const uint syncFlags) override final;

        private:
            Device::QuadLight* cu_deviceInstance = nullptr;
        };
    }
}