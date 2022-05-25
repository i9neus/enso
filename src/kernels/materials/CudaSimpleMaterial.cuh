#pragma once

#include "CudaMaterial.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class SimpleMaterial; }

    struct SimpleMaterialParams
    {
        __host__ __device__ SimpleMaterialParams();
        __host__ SimpleMaterialParams(const ::Json::Node& node, const uint flags);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ uint FromJson(const ::Json::Node& node, const uint flags);

        JitterableVec3  albedoHSV;
        JitterableVec3  incandescenceHSV;

        vec3            albedoRGB;
        vec3            incandescenceRGB;

        bool            useGrid;
    };

    namespace Device
    {
        class SimpleMaterial : public Device::Material
        {
            friend Host::SimpleMaterial;

        public:
            __device__ SimpleMaterial() : m_params() {}
            __device__ ~SimpleMaterial() {}

            SimpleMaterialParams m_params;

            __device__ virtual void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const override final;
            __device__ void Synchronise(const SimpleMaterialParams& params) { m_params = params; }
        };
    }

    namespace Host
    {
        class BxDF;

        class SimpleMaterial : public Host::Material
        {
        private:
            Device::SimpleMaterial* cu_deviceData;

        public:
            __host__ SimpleMaterial(const std::string& id, const ::Json::Node& jsonNode);
            __host__ virtual ~SimpleMaterial() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                       OnDestroyAsset() override final;
            __host__ virtual uint                       FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual Device::SimpleMaterial* GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ static std::string GetAssetTypeString() { return "simple"; }
            __host__ static std::string GetAssetDescriptionString() { return "Simple Material"; }
        };
    }
}