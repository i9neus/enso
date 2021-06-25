#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRenderObject.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host {  class Material;  }

    namespace Device
    {
        class Material : public Device::RenderObject, public AssetTags<Host::Material, Device::Material>
        {
        public:
            Material() = default;

            __device__ virtual void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const = 0;
        };
    }

    namespace Host
    {
        class Material : public Host::RenderObject, public AssetTags<Host::Tracable, Device::Tracable>
        {
        public:
            Material() = default;

            __host__ virtual Device::Material* GetDeviceInstance() const = 0;
        };
    }

    namespace Host { class SimpleMaterial; }

    struct SimpleMaterialParams
    {
        __host__ __device__ SimpleMaterialParams() : albedo(0.5f), incandescence(0.0f) {}
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

            __device__ virtual void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const override final;
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
            __host__ SimpleMaterial(const Json::Node& jsonNode);
            __host__ virtual ~SimpleMaterial() = default;

            __host__ virtual void                       OnDestroyAsset() override final;
            __host__ virtual void                       FromJson(const Json::Node& node) override final;
            __host__ virtual Device::SimpleMaterial*    GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}