#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRenderObjectContainer.cuh"

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
            __device__ __forceinline__ void SetBoundBxDF(const Device::BxDF* bxdf) { cu_bxdf = bxdf; }
            __device__ void Synchronise(const Device::BxDF* bxdf) { cu_bxdf = bxdf; }
        };
    }

    namespace Host
    {
        class Material : public Host::RenderObject, public AssetTags<Host::Material, Device::Material>
        {
        protected:
            __host__ Material(const std::string& id) : RenderObject(id) {}
            __host__ Material(const std::string& id, const ::Json::Node& node);

            std::string                             m_bxdfId;

        public:
            __host__ virtual uint                   FromJson(const ::Json::Node& node, const uint flags) override;
            __host__ virtual void                   Bind(RenderObjectContainer& objectContainer) override final;
            __host__ virtual Device::Material*      GetDeviceInstance() const = 0;

            __host__ virtual AssetType              GetAssetType() const override final { return AssetType::kMaterial; }
            __host__ static AssetType               GetAssetStaticType() { return AssetType::kMaterial; }
            __host__ static std::string             GetAssetTypeString() { return "material"; }
        };
    }
}