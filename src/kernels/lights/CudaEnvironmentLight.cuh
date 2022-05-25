#pragma once

#include "CudaLight.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class EnvironmentLight; }

    struct EnvironmentLightParams
    {
        __host__ __device__ EnvironmentLightParams() : light() {}
        __host__ EnvironmentLightParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ uint FromJson(const ::Json::Node& node, const uint flags);

        LightParams light;
    };

    namespace Device
    {
        class EnvironmentLight : public Device::Light
        {
            friend Host::EnvironmentLight;
        protected:
            vec3  m_radiance;
            float m_peakIrradiance;
            EnvironmentLightParams m_params;

        public:
            __device__ EnvironmentLight();
            __device__ ~EnvironmentLight() {}

            __device__ void Prepare();
            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec2 xi, vec3& extant, vec3& L, float& pdf) const override final;
            __device__ virtual bool Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const override final;
            __device__ virtual float Estimate(const Ray& incident, const HitCtx& hitCtx) const override final;
            __device__ virtual uchar GetLightRayFlags() const override final { return kRayDistantLightSample; }
            __device__ void Synchronise(const EnvironmentLightParams& params)
            {
                m_params = params;
                Prepare();
            }
        };
    }

    namespace Host
    {
        class EnvironmentLight : public Host::Light
        {
        private:
            Device::EnvironmentLight* cu_deviceData;
            Device::EnvironmentLight  m_hostData;

        public:
            __host__ EnvironmentLight(const std::string& id, const ::Json::Node& jsonNode);
            __host__ virtual ~EnvironmentLight() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual uint FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void OnDestroyAsset() override final;
            __host__ static std::string GetAssetTypeString() { return "environmentlight"; }
            __host__ static std::string GetAssetDescriptionString() { return "Environment Light"; }
            __host__ virtual Device::EnvironmentLight* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}