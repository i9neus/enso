#include "CudaCornellMaterial.cuh"
#include "generic/JsonUtils.h" 
#include "../bxdfs/CudaBxDF.cuh"

namespace Cuda
{
    __host__ __device__ CornellMaterialParams::CornellMaterialParams() 
    {
        for (int i = 0; i < kNumWalls; i++) { albedo[i] = 0.5f; }
    }

    __host__ CornellMaterialParams::CornellMaterialParams(const ::Json::Node& node, const uint flags) :
        CornellMaterialParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void CornellMaterialParams::ToJson(::Json::Node& node) const
    {
        node.AddArray("albedo1", std::vector<float>({ albedo[0].x, albedo[0].y, albedo[0].z }));
        node.AddArray("albedo2", std::vector<float>({ albedo[1].x, albedo[1].y, albedo[1].z }));
        node.AddArray("albedo3", std::vector<float>({ albedo[2].x, albedo[2].y, albedo[2].z }));
        node.AddArray("albedo4", std::vector<float>({ albedo[3].x, albedo[3].y, albedo[3].z }));
        node.AddArray("albedo5", std::vector<float>({ albedo[4].x, albedo[4].y, albedo[4].z }));
        node.AddArray("albedo6", std::vector<float>({ albedo[5].x, albedo[5].y, albedo[5].z }));
        
    }

    __host__ void CornellMaterialParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetVector("albedo1", albedo[0], flags);
        node.GetVector("albedo2", albedo[1], flags);
        node.GetVector("albedo3", albedo[2], flags);
        node.GetVector("albedo4", albedo[3], flags);
        node.GetVector("albedo5", albedo[4], flags);
        node.GetVector("albedo6", albedo[5], flags);        
    }

    __device__ void Device::CornellMaterial::Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const
    {
        incandescence = 0.0f;
        const int idx = clamp(int(hit.uv.x), 0, CornellMaterialParams::kNumWalls - 1);
        albedo = m_params.albedo[idx];
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
