#pragma once

#include "CudaLight.cuh"

namespace Cuda
{
    namespace Host
    {
        class SphereLight;
        class Sphere;
        class EmitterMaterial;
    }

    struct SphereLightParams
    {
        __host__ __device__ SphereLightParams();
        __host__ SphereLightParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        LightParams     light;
        vec3            radiance;
        vec3            radiantPower;
        vec3            radiantIntensity;
        float           peakRadiantIntensity;
        float           peakRadiance;
    };

    namespace Device
    {
        class SphereLight : public Device::Light
        {
            friend Host::SphereLight;
        protected:
            float                   m_discArea;
            float                   m_discRadius;
            SphereLightParams       m_params;

        public:
            __device__ SphereLight();
            __device__ virtual ~SphereLight() {}

            __device__ void Prepare();
            __device__ bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec2 xi, vec3& extant, vec3& L, float& pdf) const override final;
            __device__ bool Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const override final;
            __device__ virtual float Estimate(const Ray& incident, const HitCtx& hitCtx) const override final;

            __device__ void Synchronise(const SphereLightParams& params)
            {
                m_params = params;
                Prepare();
            }
            __device__ void Synchronise(const Objects& objects) { m_objects = objects; }
        };
    }

    namespace Host
    {
        class SphereLight : public Host::Light
        {
        private:
            Device::SphereLight* cu_deviceData;
            SphereLightParams  m_params;

            AssetHandle<Host::Sphere> m_lightSphereAsset;
            AssetHandle<Host::EmitterMaterial> m_lightMaterialAsset;

        public:
            __host__ SphereLight(const ::Json::Node& node, const std::string& id);
            __host__ virtual ~SphereLight() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void OnDestroyAsset() override final;
            __host__ virtual const RenderObjectParams* GetRenderObjectParams() const override final { return &m_params.light.renderObject; }
            __host__ static std::string GetAssetTypeString() { return "spherelight"; }
            __host__ static std::string GetAssetDescriptionString() { return "Sphere Light"; }
            __host__ virtual Device::SphereLight* GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() override final;
            __host__ virtual AssetHandle<Host::Tracable> GetTracableHandle() override final;
        };
    }
}