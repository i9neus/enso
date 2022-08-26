#include "CudaSphereLight.cuh"
#include "../tracables/CudaSphere.cuh"
#include "../materials/CudaEmitterMaterial.cuh"
#include "../math/CudaColourUtils.cuh"

#include "generic/JsonUtils.h"

#define kPeakIntensityMinThreshold 1e-6f

namespace Cuda
{
    __host__ __device__ SphereLightParams::SphereLightParams() :     
        radiance(1.0f) {}

    __host__ SphereLightParams::SphereLightParams(const ::Json::Node& node) :
        SphereLightParams()
    {
        FromJson(node, ::Json::kSilent);
    }

    __host__ void SphereLightParams::ToJson(::Json::Node& node) const
    {
        light.ToJson(node);
    }

    __host__ uint SphereLightParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        light.FromJson(node, flags);

        radiantPower = HSVToRGB(light.colourHSV()) * std::pow(2.0f, light.intensity());
        radiantIntensity = radiantPower / kFourPi;
        peakRadiantIntensity = cwiseMax(radiantIntensity);
        radiance = radiantIntensity / (light.transform.scale().x * light.transform.scale().y);
        peakRadiance = cwiseMax(radiance);

        return kRenderObjectDirtyAll;
    }

    __device__ Device::SphereLight::SphereLight()
    {
        Prepare();
    }

    __device__ void Device::SphereLight::Prepare()
    {
        m_discArea = kPi * sqr(m_params.light.transform.scale().x);
        m_discRadius = m_params.light.transform.scale().x;
    }

    __device__ float Device::SphereLight::Estimate(const Ray& incident, const HitCtx& hitCtx) const
    {
        const vec3 originDir = m_params.light.transform.trans() - hitCtx.hit.p;
        const float originDist2 = length2(originDir);

        // Shading point is inside the light
        if (originDist2 < sqr(m_discRadius)) { return 0.0f; }

        // Fast approx estimate of irradiance
        const float peakIrradiance = m_params.peakRadiantIntensity / originDist2;

        // Slower but more accurate estimate of peak irradiance
        //const float peakIrradiance = m_params.peakRadiance * kTwoPi * (1 - sqrt(originDist2 - sqr(m_discRadius)) / sqrt(originDist2));
        
        // Light is too far away to make a contribution. 
        // FIXME: This needs to be corrected in the event that we start using non-Lambertian BRDFs
        if (peakIrradiance < kPeakIntensityMinThreshold) { return 0.0f; }

        // If the entire bounding sphere of the light is below the horizon, exclude the light.
        // TODO: This is an approximation of the real visibility function, but it works okay. Find a better function later. 
        float cosTheta = dot(originDir, hitCtx.hit.n);
        return (cosTheta > 0.0f || sqr(cosTheta) < sqr(m_discRadius)) ? peakIrradiance : 0.0f;
    }

    __device__ bool Device::SphereLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec2 xi, vec3& extant, vec3& L, float& pdfLight) const
    {        
        vec3 originDir = m_params.light.transform.trans() - hitCtx.hit.p;
        float originDist = length(originDir);

        // Object inside the emitter? This is invalid, so return black.
        if (originDist <= m_discRadius)
        {
            pdfLight = 0.0f;
            return false;
        }

        originDir /= originDist;

        const float sinTheta = m_discRadius / originDist;
        originDist -= m_discRadius * sinTheta;
        const float projectedDiscRadius = m_discRadius * sqrtf(1.0f - sqr(sinTheta));

        const vec3 disc(SampleUnitDiscLowDistortion(xi) * projectedDiscRadius, originDist);
        const mat3 basis = CreateBasis(originDir);
        vec3 sampleDir = basis * disc;

        const float sampleDist = length(sampleDir);
        sampleDir /= sampleDist;

        // If the sample point is behind the shading point, invalidate the sample
        if (dot(sampleDir, hitCtx.hit.n) <= 0)
        {
            pdfLight = 0;
            return false;
        }

        const float solidAngle = kPi * sqr(projectedDiscRadius) * dot(originDir, sampleDir) / max(1e-10f, sqr(sampleDist));
        pdfLight = 1.0f / solidAngle;
        extant = sampleDir;
        L = m_params.radiance * solidAngle;
        return true;
    }

    __device__ bool Device::SphereLight::Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const
    {
        vec3 originDir = m_params.light.transform.trans() - incident.od.o;
        float originDist = length(originDir);

        // Object inside the emitter? This is invalid, so return black.
        if (originDist <= m_discRadius)
        {
            pdfLight = 0.0f;
            return false;
        }

        originDir /= originDist;

        const float sinTheta = m_discRadius / originDist;
        originDist -= m_discRadius * sinTheta;
        const float projectedDiscRadius = m_discRadius * sqrtf(1.0f - sqr(sinTheta));
       
        vec3 sampleDir = incident.od.d * originDist / dot(originDir, incident.od.d);
        const float sampleDist = length(sampleDir);
        sampleDir /= sampleDist;        

        const float solidAngle = kPi * sqr(projectedDiscRadius) * dot(originDir, sampleDir) / max(1e-10f, sqr(sampleDist));
        pdfLight = 1.0f / solidAngle;
        L = m_params.radiance;
        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::SphereLight::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLight) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::SphereLight>(id, json);
    }

    __host__  Host::SphereLight::SphereLight(const std::string& id, const ::Json::Node& jsonNode) :
        Host::Light(id, jsonNode),
        cu_deviceData(nullptr)
    {
        // Instantiate the emitter material
        m_lightMaterialAsset = CreateChildAsset<Host::EmitterMaterial>(id + "_material", kRenderObjectIsChild);
        Assert(m_lightMaterialAsset);

        // Instantiate the sphere tracable
        m_lightSphereAsset = CreateChildAsset<Host::Sphere>(id + "_planeTracable", kRenderObjectIsChild);
        Assert(m_lightSphereAsset);
        m_lightSphereAsset->SetBoundMaterialID(id + "_material");

        // Finally, instantitate the light itself 
        cu_deviceData = InstantiateOnDevice<Device::SphereLight>();
        FromJson(jsonNode, ::Json::kSilent);
    }

    __host__ void Host::SphereLight::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ uint Host::SphereLight::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Light::FromJson(node, flags);

        // Pull the parameters from the JSON dictionary and create a transform
        m_params.FromJson(node, flags);

        // Update the attributes of the child objects
        m_lightSphereAsset->UpdateParams(m_params.light.transform);
        m_lightMaterialAsset->UpdateParams(m_params.radiance);

        // Synchronise with the decvice
        SynchroniseObjects(cu_deviceData, m_params);

        return kRenderObjectDirtyAll;
    }

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::SphereLight::GetChildObjectHandles()
    {
        return std::vector<AssetHandle<Host::RenderObject>>(
            {
            m_lightSphereAsset.StaticCast<Host::RenderObject>(),
            m_lightMaterialAsset.StaticCast<Host::RenderObject>()
            }
        );
    }

    __host__ AssetHandle<Host::Tracable> Host::SphereLight::GetTracableHandle()
    {
        AssertMsgFmt(m_lightSphereAsset, "SphereLight object '%s' was not properly initialised.", GetAssetID().c_str());

        return m_lightSphereAsset.StaticCast<Host::Tracable>();
    }
}