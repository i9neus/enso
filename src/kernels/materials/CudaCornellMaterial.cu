#include "CudaCornellMaterial.cuh"
#include "generic/JsonUtils.h" 
#include "../bxdfs/CudaBxDF.cuh"
#include "../math/CudaColourUtils.cuh"

namespace Cuda
{
    __host__ __device__ CornellMaterialParams::CornellMaterialParams() 
    {
        for (int i = 0; i < kNumWalls; i++) 
        { 
            albedoRGB[i] = vec3(0.5f); 
            albedoHSV[i] = vec3(0.0f, 0.0f, 0.5f);
        }
    }

    __host__ CornellMaterialParams::CornellMaterialParams(const ::Json::Node& node, const uint flags) :
        CornellMaterialParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void CornellMaterialParams::ToJson(::Json::Node& node) const
    {       
        for (int i = 0; i < kNumWalls; i++)
        {
            albedoHSV[i].ToJson(tfm::format("albedo%i", i + 1), node);
        }
    }

    __host__ void CornellMaterialParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        for (int i = 0; i < kNumWalls; i++)
        {
            albedoHSV[i].FromJson(tfm::format("albedo%i", i + 1), node, flags);
            albedoRGB[i] = HSVToRGB(albedoHSV[i]());
        }
    }

    __device__ void Device::CornellMaterial::Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const
    {
        incandescence = 0.0f;
        const int idx = clamp(int(hit.uv.x), 0, CornellMaterialParams::kNumWalls - 1);
        albedo = m_params.albedoRGB[idx];
    }

    __host__ AssetHandle<Host::RenderObject> Host::CornellMaterial::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kMaterial) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::CornellMaterial(json), id);
    }

    __host__ Host::CornellMaterial::CornellMaterial(const ::Json::Node& node) :
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::CornellMaterial>();
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void Host::CornellMaterial::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::CornellMaterial::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::Material::FromJson(parentNode, flags);

        SynchroniseObjects(cu_deviceData, CornellMaterialParams(parentNode, flags));
    }
}
