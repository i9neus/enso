#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "generic/JsonUtils.h" 

namespace Cuda
{
    namespace Host {  class Material;  }    

    namespace Device
    {
        class Material : public Device::Asset, public AssetTags<Host::Material, Device::Material>
        {
        public:
            Material() = default;

            __device__ vec3 Evaluate(const HitCtx& hit) const;
        };
    }

    namespace Host
    {
        class Material : public Host::Asset, public AssetTags<Host::Material, Device::Material>
        {
        public:
            Material() = default;

            __host__ virtual Device::Material* GetDeviceInstance() const = 0;
        };
    }

    namespace Host { class SimpleMaterial; }

    namespace Device
    {
        class SimpleMaterial : public Device::Material
        {
            friend Host::SimpleMaterial;
        public:
            struct Params
            {
                __host__ __device__ Params() : albedo(0.5f) {}
                vec3 albedo;
            }
            m_params;

        public:
            SimpleMaterial() = default;
            ~SimpleMaterial() = default;

            __device__ vec3 Evaluate(const HitCtx& hit) const;
            __device__ void OnSyncParameters(const Params& params) { m_params = params;  }
        };
    }

    namespace Host
    {
        class SimpleMaterial : public Host::Material
        {
        private:
            Device::SimpleMaterial* cu_deviceData;

        public:
            __host__ SimpleMaterial();
            __host__ virtual ~SimpleMaterial() { OnDestroyAsset(); }

            __host__ virtual void                       OnDestroyAsset() override final;
            __host__ virtual void                       OnJson(const Json::Node& jsonNode) override final;
            __host__ virtual Device::SimpleMaterial*    GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}