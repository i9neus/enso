#pragma once

#include "CudaLight.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class EnvironmentLight; }

    struct EnvironmentLightParams
    {
        __host__ __device__ EnvironmentLightParams() : intensity(1.0f), colour(1.0f) {}
        __host__ EnvironmentLightParams(const ::Json::Node& node) { FromJson(node); }

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node);

        float intensity;
        vec3 colour;
    };

    namespace Device
    {
        class EnvironmentLight : public Device::Light
        {
            friend Host::EnvironmentLight;
        protected:
            float m_emitterArea;
            vec3  m_emitterRadiance;
            EnvironmentLightParams m_params;

        public:
            __device__ EnvironmentLight();
            __device__ ~EnvironmentLight() = default;

            __device__ void Prepare();
            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdf) const override final;
            __device__ virtual void Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const override final;
            __device__ void OnSyncParameters(const EnvironmentLightParams& params)
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
            __host__ EnvironmentLight(const ::Json::Node& jsonNode);
            __host__ virtual ~EnvironmentLight() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void FromJson(const ::Json::Node& jsonNode) override final;
            __host__ virtual void OnDestroyAsset() override final;
            __host__ static std::string GetAssetTypeString() { return "environment"; }
            __host__ virtual Device::EnvironmentLight* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}