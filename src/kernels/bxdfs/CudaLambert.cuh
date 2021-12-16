#pragma once

#include "CudaBxDF.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class LambertBRDF; }

    enum LambertProbeGridFlags : uint
    {
        kLambertGridChannel0 = 1 << 0,
        kLambertGridChannel1 = 1 << 1,
        kLambertGridChannel2 = 1 << 2,
        kLambertGridChannel3 = 1 << 3,
        kLambertUseProbeGrid = 1 << 4,
        kLambertGridNumChannels = 4
    };

    struct LambertBRDFParams
    {
        __host__ __device__ LambertBRDFParams() : probeGridFlags(kLambertUseProbeGrid | kLambertGridChannel0 | kLambertGridChannel1) { }
        __host__ LambertBRDFParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        uint    probeGridFlags;
    };

    namespace Device
    {
        class LightProbeGrid;
        
        class LambertBRDF : public Device::BxDF
        {            
            friend Host::LambertBRDF;

        public:
            struct Objects
            {
                Device::LightProbeGrid* lightProbeGrids[kLambertGridNumChannels] = { nullptr, nullptr, nullptr, nullptr };
            };

            LambertBRDF() = default;
            ~LambertBRDF() = default;

            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, const vec2& xi, vec3& extant, float& pdf) const override final;
            __device__ virtual bool Evaluate(const vec3& incident, const vec3& extant, const HitCtx& hitCtx, float& weight, float& pdf) const override final;
            __device__ virtual vec3 EvaluateCachedRadiance(const HitCtx& hitCtx) const override final;
            __device__ virtual bool IsTwoSided() const override final { return false; }

            __device__ void Synchronise(const Objects& objects) { m_objects = objects; }
            __device__ void Synchronise(const LambertBRDFParams& params) { m_params = params; }

        private:
            Device::LightProbeGrid* cu_lightProbeGrid = nullptr;
            LambertBRDFParams       m_params;
            Objects                 m_objects;
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
            std::array<AssetHandle<Host::LightProbeGrid>, kLambertGridNumChannels> m_hostLightProbeGrids;

            std::string                             m_gridIDs[kLambertGridNumChannels];
            LambertBRDFParams                       m_params;

        public:
            __host__ LambertBRDF(const ::Json::Node&);
            __host__ virtual ~LambertBRDF() { OnDestroyAsset(); }

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string&, const AssetType&, const ::Json::Node&);
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void Bind(RenderObjectContainer& sceneObjects) override final;
            __host__ virtual void OnUpdateSceneGraph(RenderObjectContainer& sceneObjects) override final;

            __host__ virtual void OnDestroyAsset() override final;
            __host__ static std::string GetAssetTypeString() { return "lambert"; }
            __host__ static std::string GetAssetDescriptionString() { return "Lambertian BRDF"; }
            __host__ virtual Device::LambertBRDF* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}