#include "CudaEnvironmentLight.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ EnvironmentLightParams::EnvironmentLightParams(const ::Json::Node& node) :
        EnvironmentLightParams()
    { 
        FromJson(node, ::Json::kRequiredWarn); 
    }
    
    __host__ void EnvironmentLightParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("intensity", intensity);
        node.AddArray("colour", std::vector<float>({ colour.x, colour.y, colour.z }));
    }

    __host__ void EnvironmentLightParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("intensity", intensity, flags);
        node.GetVector("colour", colour, flags);
    }

    __device__ Device::EnvironmentLight::EnvironmentLight()
    {
        m_emitterRadiance = vec3(1.0f);
    }

    __device__ void Device::EnvironmentLight::Prepare()
    {
        m_emitterRadiance = m_params.colour * math::pow(2.0f, m_params.intensity);
    }

    __device__ bool Device::EnvironmentLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdfLight) const
    {
        pdfLight = 1 / kFourPi;
        extant = SampleUnitSphere(renderCtx.Rand<0, 1>());
    }

    __device__ void Device::EnvironmentLight::Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const
    {
        const float solidAngle = dot(hitCtx.hit.n, incident.od.d) * m_emitterArea / sqr(incident.tNear);

        pdfLight = 1 / solidAngle;
        L = m_emitterRadiance;
    }

    __host__ AssetHandle<Host::RenderObject> Host::EnvironmentLight::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::EnvironmentLight(json), id);
    }

    __host__  Host::EnvironmentLight::EnvironmentLight(const ::Json::Node& jsonNode)
        : cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::EnvironmentLight>();
        FromJson(jsonNode, ::Json::kRequiredWarn);
    }

    __host__ void Host::EnvironmentLight::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::EnvironmentLight::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Json::Node childNode = parentNode.GetChildObject("material", flags);
        if (childNode)
        {
            SynchroniseObjects(cu_deviceData, EnvironmentLightParams(childNode));
        }
    }
}