#pragma once

#include "CudaLight.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class DistantLight; }

    struct DistantLightParams
    {
        __host__ __device__ DistantLightParams();
        __host__ DistantLightParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        LightParams light;
        float angle;
    };

    namespace Device
    {
        class DistantLight : public Device::Light
        {
            friend Host::DistantLight;
        protected:
            DistantLightParams m_params;

            float   m_discRadius;
            float   m_discArea;
            mat3    m_basis;
            vec3    m_radiance;
            float   m_peakIrradiance;
            float   m_cosAngle;
            float   m_solidAngle;

        public:
            __device__ DistantLight();
            __device__ ~DistantLight() {}

            __device__ void Prepare();
            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec2 xi, vec3& extant, vec3& L, float& pdf) const override final;
            __device__ virtual bool Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const override final;
            __device__ virtual float Estimate(const Ray& incident, const HitCtx& hitCtx) const override final;
            __device__ virtual uchar GetLightRayFlags() const override final { return kRayDistantLightSample; }

            __device__ void Synchronise(const DistantLightParams& params)
            {
                m_params = params;
                Prepare();
            }
        };
    }

    namespace Host
    {
        class DistantLight : public Host::Light
        {
        private:
            Device::DistantLight* cu_deviceData;
            Device::DistantLight  m_hostData;

        public:
            __host__ DistantLight(const std::string& id, const ::Json::Node& jsonNode);
            __host__ virtual ~DistantLight() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void OnDestroyAsset() override final;
            __host__ static std::string GetAssetTypeString() { return "distantlight"; }
            __host__ static std::string GetAssetDescriptionString() { return "Distant Light"; }
            __host__ virtual Device::DistantLight* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}