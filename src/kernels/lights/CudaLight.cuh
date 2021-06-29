#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRenderObject.cuh"

namespace Cuda
{
    namespace Host { class Light; }

    namespace Device
    {
        class Light : public Device::RenderObject, public AssetTags<Host::Light, Device::Light>
        {
        public:
            __device__ Light() {}
            __device__ virtual ~Light() {}

            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdf) const = 0;
            __device__ virtual void Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const = 0;
        };
    }

    namespace Host
    {
        class Light : public Host::RenderObject, public AssetTags<Host::Light, Device::Light>
        {
        public:
            __host__ virtual Device::Light* GetDeviceInstance() const = 0;
            __host__ virtual AssetType GetAssetType() const override final { return AssetType::kLight; }
            __host__ static std::string GetAssetTypeString() { return "light"; }
        };
    }
}