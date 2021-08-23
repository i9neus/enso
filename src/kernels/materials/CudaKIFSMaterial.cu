﻿#include "CudaKIFSMaterial.cuh"
#include "generic/JsonUtils.h" 
#include "../bxdfs/CudaBxDF.cuh"
#include "../math/CudaColourUtils.cuh"

namespace Cuda
{
    __host__ __device__ KIFSMaterialParams::KIFSMaterialParams() :
        incandescenceHSV(vec3(0.0f)),
        albedoHSV(vec3(0.0f, 0.0f, 0.7f)),
        incandescenceRGB(vec3(0.0f)),
        albedoRGB(0.0f)
    {}

    __host__ KIFSMaterialParams::KIFSMaterialParams(const ::Json::Node& node, const uint flags) :
        KIFSMaterialParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void KIFSMaterialParams::ToJson(::Json::Node& node) const
    {
        incandescenceHSV.ToJson("incandescence", node);
        albedoHSV.ToJson("albedo", node);
    }

    __host__ void KIFSMaterialParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        incandescenceHSV.FromJson("incandescence", node, flags);
        albedoHSV.FromJson("albedo", node, flags);
        
        incandescenceRGB = HSVToRGB(incandescenceHSV());
        albedoRGB = HSVToRGB(albedoHSV());
    }

    __device__ void Device::KIFSMaterial::Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const
    {
        const uint code = HashOf(*reinterpret_cast<const uint*>(&hit.uv.x));

        vec3 alpha((code & ((1 << 10) - 1)) / float((1 << 10) - 1),
                    ((code >> 10) & ((1 << 10) - 1)) / float((1 << 10) - 1),
                    ((code >> 20) & ((1 << 10) - 1)) / float((1 << 10) - 1));

        incandescence = m_params.incandescenceRGB;
        albedo = m_params.albedoRGB;
    }

    __host__ AssetHandle<Host::RenderObject> Host::KIFSMaterial::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kMaterial) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::KIFSMaterial(json), id);
    }

    __host__ Host::KIFSMaterial::KIFSMaterial(const ::Json::Node& node) :
        cu_deviceData(nullptr)
    {
        RenderObject::SetRenderObjectFlags(kIsJitterable);
        
        cu_deviceData = InstantiateOnDevice<Device::KIFSMaterial>();
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void Host::KIFSMaterial::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::KIFSMaterial::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::Material::FromJson(parentNode, flags);

        SynchroniseObjects(cu_deviceData, KIFSMaterialParams(parentNode, flags));
    }
}
