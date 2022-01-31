#pragma once

#include "CudaMaterial.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class CornellMaterial; }

    struct CornellMaterialParams
    {
        enum _attrs : int { kNumWalls = 6 };

        __host__ __device__ CornellMaterialParams();
        __host__ CornellMaterialParams(const ::Json::Node& node, const uint flags);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        JitterableVec3    albedoHSV[kNumWalls];
        vec3              albedoRGB[6];
    };

    namespace Device
    {
        class CornellMaterial : public Device::Material
        {
            friend Host::CornellMaterial;

        public:
            __device__ CornellMaterial() : m_params() {}
            __device__ ~CornellMaterial() {}

            CornellMaterialParams m_params;

            __device__ virtual void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const override final;
            __device__ void Synchronise(const CornellMaterialParams& params) { m_params = params; }
        };
    }

    namespace Host
    {
        class BxDF;

        class CornellMaterial : public Host::Material
        {
        private:
            Device::CornellMaterial* cu_deviceData;
            CornellMaterialParams    m_params;

        public:
            __host__ CornellMaterial(const std::string& id, const ::Json::Node& jsonNode);
            __host__ virtual ~CornellMaterial() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                       OnDestroyAsset() override final;
            __host__ virtual void                       FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual Device::CornellMaterial*   GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ static std::string                 GetAssetTypeString() { return "cornellmaterial"; }
            __host__ static std::string                 GetAssetDescriptionString() { return "Cornell Box Material"; }
        };
    }
}