#include "CudaEnvironmentLight.cuh"
#include "generic/JsonUtils.h"

#include "../math/CudaColourUtils.cuh"

namespace Cuda
{
    __host__ EnvironmentLightParams::EnvironmentLightParams(const ::Json::Node& node) :
        EnvironmentLightParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void EnvironmentLightParams::ToJson(::Json::Node& node) const
    {
        light.ToJson(node);
    }

    __host__ void EnvironmentLightParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        light.FromJson(node, flags);
    }

    __device__ Device::EnvironmentLight::EnvironmentLight()
    {
        m_radiance = vec3(1.0f);
    }

    __device__ void Device::EnvironmentLight::Prepare()
    {
        m_radiance = HSVToRGB(m_params.light.colourHSV()) * std::pow(2.0f, m_params.light.intensity());
        m_peakIrradiance = kTwoPi * cwiseMax(m_radiance);
    }

    __device__ float Device::EnvironmentLight::Estimate(const Ray& incident, const HitCtx& hitCtx) const
    {
        return m_peakIrradiance;
    }

    __device__ bool Device::EnvironmentLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec2 xi, vec3& extant, vec3& L, float& pdfLight) const
    {
        pdfLight = 1 / kTwoPi;
        extant = CreateBasis(hitCtx.hit.n) * SampleUnitHemisphere(renderCtx.rng.Rand<0, 1>());
        L = m_radiance* kTwoPi;

        return true;
    }

    __device__ bool Device::EnvironmentLight::Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const
    {        
        pdfLight = 1 / kTwoPi;
        L = m_radiance;
        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::EnvironmentLight::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLight) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::EnvironmentLight>(id, json);
    }

    __host__  Host::EnvironmentLight::EnvironmentLight(const std::string& id, const ::Json::Node& jsonNode) :
        Host::Light(id, jsonNode),
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::EnvironmentLight>(id);
        FromJson(jsonNode, ::Json::kRequiredWarn);
    }

    __host__ void Host::EnvironmentLight::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __host__ void Host::EnvironmentLight::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::Light::FromJson(parentNode, flags);

        SynchroniseObjects(cu_deviceData, EnvironmentLightParams(parentNode));
    }
}