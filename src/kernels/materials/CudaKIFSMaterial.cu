#include "CudaKIFSMaterial.cuh"
#include "generic/JsonUtils.h" 
#include "../bxdfs/CudaBxDF.cuh"
#include "../math/CudaColourUtils.cuh"

namespace Cuda
{
    __host__ __device__ KIFSMaterialParams::KIFSMaterialParams() :
        incandescenceHSV(vec3(0.0f)),
        albedoHSV(vec3(0.0f, 0.0f, 0.7f)),
        incandescenceRGB(vec3(0.0f))
    {
        albedoHSVRange[0] = albedoHSVRange[1] = vec3(0.0f, 0.0f, 1.0f);
    }

    __host__ KIFSMaterialParams::KIFSMaterialParams(const ::Json::Node& node, const uint flags) :
        KIFSMaterialParams()
    {
        FromJson(node, ::Json::kSilent);
    }

    __host__ void KIFSMaterialParams::ToJson(::Json::Node& node) const
    {
        incandescenceHSV.ToJson("incandescence", node);
        albedoHSV.ToJson("albedo", node);
    }

    __host__ uint KIFSMaterialParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        incandescenceHSV.FromJson("incandescence", node, flags);
        albedoHSV.FromJson("albedo", node, flags);
        albedoHSVRange[0] = albedoHSV.p - albedoHSV.dpdt;
        albedoHSVRange[1] = albedoHSV.p + albedoHSV.dpdt;
        
        incandescenceRGB = HSVToRGB(incandescenceHSV());

        return kRenderObjectDirtyAll;
    }

    __device__ void Device::KIFSMaterial::Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const
    {
        const uint code = HashOf(*reinterpret_cast<const uint*>(&hit.uv.x));

        vec3 alpha((code & ((1 << 10) - 1)) / float((1 << 10) - 1),
                    ((code >> 10) & ((1 << 10) - 1)) / float((1 << 10) - 1),
                    ((code >> 20) & ((1 << 10) - 1)) / float((1 << 10) - 1));

        incandescence = m_params.incandescenceRGB;
        albedo =  HSVToRGB(cwiseMix(m_params.albedoHSVRange[0], m_params.albedoHSVRange[1], fmod(vec3(alpha) + m_params.albedoHSV.t, kOne)));
    }

    __host__ AssetHandle<Host::RenderObject> Host::KIFSMaterial::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kMaterial) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::KIFSMaterial>(id, json);
    }

    __host__ Host::KIFSMaterial::KIFSMaterial(const std::string& id, const ::Json::Node& node) :
        Material(id, node),
        cu_deviceData(nullptr)
    {        
        cu_deviceData = InstantiateOnDevice<Device::KIFSMaterial>(id);
        FromJson(node, ::Json::kSilent);
    }

    __host__ void Host::KIFSMaterial::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __host__ uint Host::KIFSMaterial::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::Material::FromJson(parentNode, flags);

        SynchroniseObjects(cu_deviceData, KIFSMaterialParams(parentNode, flags));

        return kRenderObjectDirtyAll;
    }
}
