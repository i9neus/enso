#include "CudaMaterial.cuh"
#include "generic/JsonUtils.h" 

namespace Cuda
{
    __host__ void SimpleMaterialParams::ToJson(::Json::Node& node) const
    {
        node.AddArray("albedo", std::vector<float>({ albedo.x, albedo.y, albedo.z }));
        node.AddArray("incandescence", std::vector<float>({ incandescence.x, incandescence.y, incandescence.z }));
    }

    __host__ void SimpleMaterialParams::FromJson(const ::Json::Node& node)
    {
        node.GetVector("albedo", albedo, true);
        node.GetVector("incandescence", incandescence, true);
    }
    
    __device__ void Device::SimpleMaterial::Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const
    {
        constexpr float kGridScale = 5.0f;

        incandescence = m_params.incandescence;
        albedo = m_params.albedo;

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

    __host__ AssetHandle<Host::RenderObject> Host::SimpleMaterial::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::SimpleMaterial(json), id);
    }
    
    __host__ Host::SimpleMaterial::SimpleMaterial(const ::Json::Node& node) :
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::SimpleMaterial>();
        FromJson(node);
    }

    __host__ void Host::SimpleMaterial::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }

    __host__ void Host::SimpleMaterial::FromJson(const ::Json::Node& parentNode)
    {
        Json::Node childNode = parentNode.GetChildObject("material", true);
        if (childNode)
        {
            SyncParameters(cu_deviceData, SimpleMaterialParams(childNode));
        }
    }
}
    