#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRenderObject.cuh"

namespace Cuda
{
    namespace Host { class BxDF; }

    namespace Device
    {
        class BxDF : public Device::RenderObject, public AssetTags<Host::BxDF, Device::BxDF>
        {
        public:
            BxDF() = default;

            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, float& pdf) const = 0;
            __device__ virtual bool Evaluate(const vec3& incident, const vec3& extant, const HitCtx& hitCtx, float& weight, float& pdf) const = 0;

        protected:
            __device__ ~BxDF() = default;
        };
    }

    namespace Host
    {
        class BxDF : public Host::RenderObject, public AssetTags<Host::BxDF, Device::BxDF>
        {
        public:
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override
            {
                Host::RenderObject::FromJson(node, flags);
            }
            __host__ virtual Device::BxDF* GetDeviceInstance() const = 0;
            __host__ virtual AssetType GetAssetType() const override final { return AssetType::kBxDF; }
            __host__ static std::string GetAssetTypeString() { return "BxDF"; }
        };
    }
}