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
        class BxDF;
        
        class Material : public Device::RenderObject, public AssetTags<Host::Material, Device::Material>
        {
        protected:
            const Device::BxDF* cu_bxdf;

        public:
            __device__ Material() : cu_bxdf(nullptr) {}

            __device__ virtual void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const = 0;
            __device__ __forceinline__ const Device::BxDF* GetBoundBxDF() const { return cu_bxdf; }
        };
    }

    namespace Host
    {
        class Material : public Host::RenderObject, public AssetTags<Host::Material, Device::Material>
        {
        protected:
            __host__ Material() = default;

        public:
            __host__ virtual Device::Material* GetDeviceInstance() const = 0;
            __host__ virtual AssetType GetAssetType() const override final { return AssetType::kMaterial; }
            __host__ static std::string GetAssetTypeString() { return "material"; }
        };
    }

    namespace Host { class SimpleMaterial; }

    struct SimpleMaterialParams
    {
        __host__ __device__ SimpleMaterialParams() : albedo(0.5f), incandescence(0.0f) {}
        __host__ SimpleMaterialParams(const ::Json::Node& node, const uint flags);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        vec3            albedo;
        vec3            incandescence;
    };

    namespace Device
    {
        class SimpleMaterial : public Device::Material
        {
            friend Host::SimpleMaterial;

        public:
            __device__ SimpleMaterial() : m_params() {}
            __device__ ~SimpleMaterial() {}

            SimpleMaterialParams m_params;

            __device__ virtual void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const override final;
            __device__ void Synchronise(const SimpleMaterialParams& params) { m_params = params;  }
            __device__ void Synchronise(const Device::BxDF* bxdf) { cu_bxdf = bxdf; }
        };
    }

    namespace Host
    {
        class BxDF;

        class SimpleMaterial : public Host::Material
        {
        private:
            Device::SimpleMaterial*     cu_deviceData;

            std::string                 m_bxdfId;

        public:
            __host__ SimpleMaterial(const ::Json::Node& jsonNode);
            __host__ virtual ~SimpleMaterial() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                       OnDestroyAsset() override final;
            __host__ virtual void                       FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void                       Bind(RenderObjectContainer& objectContainer) override final;
            __host__ virtual Device::SimpleMaterial*    GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ static std::string GetAssetTypeString() { return "simple"; }
        };
    }
}