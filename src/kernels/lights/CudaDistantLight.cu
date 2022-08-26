#include "CudaDistantLight.cuh"
#include "generic/JsonUtils.h"

#include "../math/CudaColourUtils.cuh"

namespace Cuda
{
    __host__ __device__ DistantLightParams::DistantLightParams() : angle(0.53f)
    {

    }
    
    __host__ DistantLightParams::DistantLightParams(const ::Json::Node& node) :
        DistantLightParams()
    {
        FromJson(node, ::Json::kSilent);
    }

    __host__ void DistantLightParams::ToJson(::Json::Node& node) const
    {
        light.ToJson(node);
        node.AddValue("angle", angle);
    }

    __host__ uint DistantLightParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        light.FromJson(node, flags);
        node.GetValue("angle", angle, flags);

        angle = toRad(clamp(angle, 0.0f, 80.0f));

        return kRenderObjectDirtyAll;
    }

    __device__ Device::DistantLight::DistantLight() : m_radiance(1.0f)
    {
    }

    __device__ void Device::DistantLight::Prepare()
    {
        m_radiance = HSVToRGB(m_params.light.colourHSV()) * std::pow(2.0f, m_params.light.intensity());
        m_peakIrradiance = cwiseMax(m_radiance);

        m_discRadius = tan(m_params.angle);
        m_discArea = kPi * sqr(m_discRadius);        
        m_basis = transpose(m_params.light.transform.inv);
        m_cosAngle = cos(m_params.angle);
        // TODO: Analytically sample the solid angle
        m_solidAngle = kTwoPi * (1 - cos(m_params.angle));
    }

    __device__ bool Device::DistantLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec2 xi, vec3& extant, vec3& L, float& pdfLight) const
    {
        const vec2 disc = SampleUnitDiscLowDistortion(xi);
        vec3 sampleDir = m_basis.x * disc.x * m_discRadius +
                         m_basis.y * disc.y * m_discRadius +
                         m_basis.z;

        // If the sample point is behind the shading point, invalidate the sample
        if (dot(sampleDir, hitCtx.hit.n) <= 0) { return false; }

        float sampleDist2 = length2(sampleDir);
        sampleDir /= sqrtf(sampleDist2);
        const float solidAngle = m_discArea * dot(m_basis.z, sampleDir) / sampleDist2;

        pdfLight = 1.0f / solidAngle;
        extant = sampleDir;
        L = m_radiance;
        return true;
    }

    __device__ bool Device::DistantLight::Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const
    {
        const float theta = acos(dot(incident.od.d, m_basis.z));
        if (theta > m_params.angle)
        {
            return false;
        }

        pdfLight = 1.0f / m_solidAngle;
        L = m_radiance;
        return true;
    }

    __device__ float Device::DistantLight::Estimate(const Ray& incident, const HitCtx& hitCtx) const
    {
        // If the entire bounding sphere of the light is below the horizon, exclude the light.
        // TODO: This is an approximation of the real visibility function, but it works okay. Find a better function later. 
        float cosTheta = dot(m_basis.z, hitCtx.hit.n);
        return (cosTheta > 0.0f || sqr(cosTheta) < sqr(m_discRadius)) ? m_peakIrradiance : 0.0f;
    }

    __host__ AssetHandle<Host::RenderObject> Host::DistantLight::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLight) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::DistantLight>(id, json);
    }

    __host__  Host::DistantLight::DistantLight(const std::string& id, const ::Json::Node& jsonNode) :
        Host::Light(id, jsonNode),
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::DistantLight>();
        FromJson(jsonNode, ::Json::kSilent);
    }

    __host__ void Host::DistantLight::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ uint Host::DistantLight::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::Light::FromJson(parentNode, flags);

        SynchroniseObjects(cu_deviceData, DistantLightParams(parentNode));

        return kRenderObjectDirtyAll;
    }
}