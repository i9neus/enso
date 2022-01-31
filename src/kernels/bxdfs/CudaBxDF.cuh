#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRenderObjectContainer.cuh"

namespace Cuda
{
    namespace Host { class BxDF; }

    namespace Device
    {
        class BxDF : public Device::RenderObject, public AssetTags<Host::BxDF, Device::BxDF>
        {
        public:
            BxDF() = default;

            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, const vec2& xi, vec3& extant, float& pdf) const = 0;
            __device__ virtual bool Evaluate(const vec3& incident, const vec3& extant, const HitCtx& hitCtx, float& weight, float& pdf) const = 0;
            __device__ virtual vec3 EvaluateCachedRadiance(const HitCtx& hitCtx) const { return vec3(0.0f); }
            __device__ virtual bool IsTwoSided() const = 0;

        protected:
            __device__ ~BxDF() = default;
        };
    }

    namespace Host
    {
        class BxDF : public Host::RenderObject, public AssetTags<Host::BxDF, Device::BxDF>
        {
        public:
            BxDF(const std::string& id, const ::Json::Node& node) : Host::RenderObject(id)
            {
                Host::RenderObject::UpdateDAGPath(node);
            }

            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override
            {
                
            }
            __host__ virtual Device::BxDF*              GetDeviceInstance() const = 0;
            __host__ virtual AssetType                  GetAssetType() const override final { return AssetType::kBxDF; }
            __host__ static AssetType                   GetAssetStaticType() { return AssetType::kBxDF; }
            __host__ static std::string                 GetAssetTypeString() { return "BxDF"; }
        };
    }
}