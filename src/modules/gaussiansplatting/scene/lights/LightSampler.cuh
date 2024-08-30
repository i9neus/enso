#pragma once

#include "../tracables/Tracable.cuh"

namespace Enso
{        
    struct LightSamplerParams
    {
        __device__ void Validate() const {}

        BidirectionalTransform transform;
        vec3 radiance;
    };
    
    namespace Device
    {
        class LightSampler : public Device::SceneObject
        {
        public:
            __device__      LightSampler() {}
            __device__      ~LightSampler() noexcept {}

            __device__ virtual float Sample(const Ray& incident, Ray& extant, const HitCtx& hit, const vec2& xi) const = 0;
            __device__ virtual float Evaluate(Ray& extant, const HitCtx& hit) const = 0;

            __device__ void Synchronise(const LightSamplerParams& params) { m_params = params; }

        protected:
            LightSamplerParams m_params;
        };
    }

    namespace Host
    {
        class LightSampler : public Host::SceneObject
        {
        public:
            __host__  LightSampler(const InitCtx& initCtx);
            __host__  virtual ~LightSampler() noexcept {}

            __host__ Device::LightSampler* GetDeviceInstance() const { return cu_deviceInstance; }
            __host__ bool BindTracable(AssetHandle<Host::Tracable>& tracable);

            __host__ virtual void Synchronise(const uint syncFlags) override final;

        protected:
            __host__ virtual void OnDirty(const DirtinessEvent& flag, AssetHandle<Host::Asset>& caller) override final;
            __host__ virtual bool TryBind(AssetHandle<Host::Tracable>& tracable) = 0;

            __host__ void SetDeviceInstance(Device::LightSampler* deviceInstance) { cu_deviceInstance = deviceInstance; }

        protected:
            LightSamplerParams      m_params;

        private:
            Device::LightSampler*   cu_deviceInstance;
            WeakAssetHandle<Host::Tracable> m_weakTracable;
        };
    }
}