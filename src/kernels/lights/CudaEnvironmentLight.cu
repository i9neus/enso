#include "CudaEnvironmentLight.cuh"

namespace Cuda
{
    __host__ void EnvironmentLightParams::ToJson(Json::Node& node) const
    {
        node.AddValue("intensity", intensity);
        node.AddArray("colour", std::vector<float>({ colour.x, colour.y, colour.z }));
    }

    __host__ void EnvironmentLightParams::FromJson(const Json::Node& node)
    {
        node.GetValue("intensity", intensity, true);
        node.GetVector("colour", colour, true);
    }

    __device__ Device::EnvironmentLight::EnvironmentLight(const BidirectionalTransform& transform) : Light(transform)
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

    __host__  Host::EnvironmentLight::EnvironmentLight()
        : cu_deviceData(nullptr)
    {
        m_hostData.m_transform.MakeIdentity();

        cu_deviceData = InstantiateOnDevice<Device::EnvironmentLight>(m_hostData.m_transform);
    }

    __host__ void Host::EnvironmentLight::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }

    __host__ void Host::EnvironmentLight::OnJson(const Json::Node& parentNode)
    {
        Json::Node childNode = parentNode.GetChildObject("material", true);
        if (childNode)
        {
            SyncParameters(cu_deviceData, EnvironmentLightParams(childNode));
        }
    }
}