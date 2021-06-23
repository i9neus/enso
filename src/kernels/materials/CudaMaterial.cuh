#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host {  class Material;  }    

    namespace Device
    {
        class Material : public Device::Asset, public AssetTags<Host::Material, Device::Material>
        {
        public:
            Material() = default;

            __device__ void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const;
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

    struct SimpleMaterialParams
    {
        __host__ __device__ SimpleMaterialParams() : albedo(0.5f) {}
        __host__ SimpleMaterialParams(const Json::Node& node) { FromJson(node); }

        __host__ void ToJson(Json::Node& node) const;
        __host__ void FromJson(const Json::Node& node);

        vec3 albedo;
        vec3 incandescence;
    };

    namespace Device
    {
        class SimpleMaterial : public Device::Material
        {
            friend Host::SimpleMaterial;

        public:
            __host__ __device__ SimpleMaterial() : m_params() {}
            ~SimpleMaterial() = default;

            SimpleMaterialParams m_params;

            __device__ void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const;
            __device__ void OnSyncParameters(const SimpleMaterialParams& params) { m_params = params;  }
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