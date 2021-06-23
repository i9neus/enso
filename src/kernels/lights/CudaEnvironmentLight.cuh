#pragma once

#include "CudaLight.cuh"

namespace Cuda
{
    namespace Host { class EnvironmentLight; }

    struct EnvironmentLightParams
    {
        __host__ __device__ EnvironmentLightParams() : intensity(1.0f), colour(1.0f) {}
        __host__ EnvironmentLightParams(const Json::Node& node) { FromJson(node); }

        __host__ void ToJson(Json::Node& node) const;
        __host__ void FromJson(const Json::Node& node);

        float intensity;
        vec3 colour;
    };

    namespace Device
    {
        class EnvironmentLight : public Device::Light
        {
            friend Host::EnvironmentLight;
        protected:
            EnvironmentLight() = default;

            float m_emitterArea;
            vec3  m_emitterRadiance;
            EnvironmentLightParams m_params;

        public:
            __device__ EnvironmentLight(const BidirectionalTransform& transform);
            __device__ ~EnvironmentLight() = default;

            __device__ void Prepare();
            __device__ bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdf) const;
            __device__ void Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const;
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
            __host__ EnvironmentLight();
            __host__ virtual ~EnvironmentLight() { OnDestroyAsset(); }
            __host__ virtual void OnJson(const Json::Node& jsonNode) override final;
            __host__ virtual void OnDestroyAsset() override final;

            __host__ virtual Device::EnvironmentLight* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}