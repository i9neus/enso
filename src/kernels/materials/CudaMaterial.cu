#include "CudaMaterial.cuh"
#include "generic/JsonUtils.h" 

namespace Cuda
{
    __host__ void Device::SimpleMaterial::Params::ToJson(Json::Node& node) const
    {
        node.AddArray("albedo", std::vector<float>({ albedo.x, albedo.y, albedo.z }));
    }

    __host__ void Device::SimpleMaterial::Params::FromJson(const Json::Node& node)
    {
        node.GetVector("albedo", albedo, true);
    }
    
    __device__ vec3 Device::SimpleMaterial::Evaluate(const HitCtx& hit) const
    {
        constexpr float kGridScale = 5.0f;

        vec3 albedo = m_params.albedo;
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

        return albedo;
    }
    
    __host__ Host::SimpleMaterial::SimpleMaterial() :
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::SimpleMaterial>();
    }

    __host__ void Host::SimpleMaterial::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }

    __host__ void Host::SimpleMaterial::OnJson(const Json::Node& parentNode)
    {
        Json::Node childNode = parentNode.GetChildObject("material", true);
        if (childNode)
        {
            SyncParameters(cu_deviceData, Device::SimpleMaterial::Params(childNode));
        }
    }
}
    