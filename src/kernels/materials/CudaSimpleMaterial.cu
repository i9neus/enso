#include "CudaSimpleMaterial.cuh"
#include "generic/JsonUtils.h" 
#include "../bxdfs/CudaBxDF.cuh"
#include "../math/CudaColourUtils.cuh"

namespace Cuda
{
    __host__ __device__ SimpleMaterialParams::SimpleMaterialParams() :
        incandescenceHSV(vec3(0.0f)),
        albedoHSV(vec3(0.0f, 0.0f, 0.7f)),
        incandescenceRGB(HSVToRGB(incandescenceHSV())),
        albedoRGB(HSVToRGB(albedoHSV())),
        useGrid(false) {}

    __host__ SimpleMaterialParams::SimpleMaterialParams(const ::Json::Node& node, const uint flags) :
        SimpleMaterialParams()
    {
        FromJson(node, ::Json::kSilent);
    }

    __host__ void SimpleMaterialParams::ToJson(::Json::Node& node) const
    {
        incandescenceHSV.ToJson("incandescence", node);
        albedoHSV.ToJson("albedo", node);
        node.AddValue("useGrid", useGrid);
    }

    __host__ uint SimpleMaterialParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        incandescenceHSV.FromJson("incandescence", node, flags);
        albedoHSV.FromJson("albedo", node, flags);
        node.GetValue("useGrid", useGrid, flags);

        incandescenceRGB = HSVToRGB(incandescenceHSV());
        albedoRGB = HSVToRGB(albedoHSV());

        return kRenderObjectDirtyAll;
    }

    __device__ void Device::SimpleMaterial::Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const
    {
        constexpr float kGridScale = 5.0f;

        incandescence = m_params.incandescenceRGB;
        albedo = m_params.albedoRGB;

        if (m_params.useGrid)
        {
            vec2 absUv = abs(hit.uv - vec2(0.5f));
            if (absUv.x < 0.52f && absUv.y < 0.52f && !(absUv.x < 0.5f && absUv.y < 0.5f))
            {
                albedo *= 0.7;
            }
            if (fract(absUv.x * kGridScale) < 0.02f || fract(absUv.y * kGridScale) < 0.02f ||
                fract(absUv.x * 10.0f * kGridScale) < 0.1f || fract(absUv.y * 10.0 * kGridScale) < 0.1f)
            {
                albedo *= 0.7;
            }
        }
    }

    __host__ AssetHandle<Host::RenderObject> Host::SimpleMaterial::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kMaterial) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::SimpleMaterial>(id, json);
    }

    __host__ Host::SimpleMaterial::SimpleMaterial(const std::string& id, const ::Json::Node& node) :
        Material(id, node),
        cu_deviceData(nullptr)
    {       
        cu_deviceData = InstantiateOnDevice<Device::SimpleMaterial>();

        FromJson(node, ::Json::kSilent);
    }

    __host__ void Host::SimpleMaterial::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ uint Host::SimpleMaterial::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::Material::FromJson(parentNode, flags);

        SynchroniseObjects(cu_deviceData, SimpleMaterialParams(parentNode, flags));

        return kRenderObjectDirtyAll;
    }
}
