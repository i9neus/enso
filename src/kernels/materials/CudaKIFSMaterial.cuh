#pragma once

#include "CudaMaterial.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class KIFSMaterial; }

    struct KIFSMaterialParams
    {
        __host__ __device__ KIFSMaterialParams();
        __host__ KIFSMaterialParams(const ::Json::Node& node, const uint flags);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        JitterableVec3  albedoHSV;
        vec3            albedoHSVRange[2];

        JitterableVec3  incandescenceHSV;
        vec3            incandescenceRGB;
    };

    namespace Device
    {
        class KIFSMaterial : public Device::Material
        {
            friend Host::KIFSMaterial;

        public:
            __device__ KIFSMaterial() : m_params() {}
            __device__ ~KIFSMaterial() {}

            KIFSMaterialParams m_params;

            __device__ virtual void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const override final;
            __device__ void Synchronise(const KIFSMaterialParams& params) { m_params = params; }
        };
    }

    namespace Host
    {
        class BxDF;

        class KIFSMaterial : public Host::Material
        {
        private:
            Device::KIFSMaterial* cu_deviceData;

        public:
            __host__ KIFSMaterial(const ::Json::Node& jsonNode);
            __host__ virtual ~KIFSMaterial() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                       OnDestroyAsset() override final;
            __host__ virtual void                       FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual Device::KIFSMaterial* GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ static std::string GetAssetTypeString() { return "kifsmaterial"; }
            __host__ static std::string GetAssetDescriptionString() { return "KIFS Material"; }
        };
    }
}