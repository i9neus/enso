#pragma once

#include "CudaBxDF.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class LambertBRDF; }

    namespace Device
    {
        class LightProbeGrid;
        
        class LambertBRDF : public Device::BxDF
        {            
            friend Host::LambertBRDF;

        public:
            LambertBRDF() = default;
            ~LambertBRDF() = default;

            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, float& pdf) const override final;
            __device__ virtual bool Evaluate(const vec3& incident, const vec3& extant, const HitCtx& hitCtx, float& weight, float& pdf) const override final;
            __device__ virtual vec3 EvaluateCachedRadiance(const HitCtx& hitCtx) const override final;

            __device__ void Synchronise(Device::LightProbeGrid* lightProbeGrid) { cu_lightProbeGrid = lightProbeGrid; }

        private:
            Device::LightProbeGrid* cu_lightProbeGrid = nullptr;
        };
    }

    namespace Host
    {
        class LightProbeGrid;
        
        class LambertBRDF : public Host::BxDF
        {
        private:
            Device::LambertBRDF*                    cu_deviceData;
            Device::LambertBRDF                     m_hostData; 
            AssetHandle<Host::LightProbeGrid>	    m_hostLightProbeGrid;

            std::string         m_lightProbeGridID;

        public:
            __host__ LambertBRDF(const ::Json::Node&);
            __host__ virtual ~LambertBRDF() { OnDestroyAsset(); }

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string&, const AssetType&, const ::Json::Node&);
            __host__ virtual void Bind(RenderObjectContainer& sceneObjects) override final;

            __host__ virtual void OnDestroyAsset() override final;
            __host__ static std::string GetAssetTypeString() { return "lambert"; }
            __host__ static std::string GetAssetDescriptionString() { return "Lambertian BRDF"; }
            __host__ virtual Device::LambertBRDF* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}