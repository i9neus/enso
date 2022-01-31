#include "CudaQuadLight.cuh"
#include "../tracables/CudaPlane.cuh"
#include "../materials/CudaEmitterMaterial.cuh"
#include "../math/CudaColourUtils.cuh"

#include "generic/JsonUtils.h"

#define kPeakIntensityMinThreshold 1e-6f

namespace Cuda
{
    __host__ __device__ QuadLightParams::QuadLightParams() :
        light(),
        radiance(0.0f) {}
    
    __host__ QuadLightParams::QuadLightParams(const ::Json::Node& node) :
        QuadLightParams()
    { 
         FromJson(node, ::Json::kRequiredWarn); 
    }
    
    __host__ void QuadLightParams::ToJson(::Json::Node& node) const
    {
        light.ToJson(node);
    }

    __host__ void QuadLightParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        light.FromJson(node, flags);

        radiantPower = HSVToRGB(light.colourHSV()) * std::pow(2.0f, light.intensity());
        radiantIntensity = radiantPower / kPi;
        peakRadiantIntensity = cwiseMax(radiantIntensity);
        radiance = radiantIntensity / (light.transform.scale().x * light.transform.scale().y);
        peakRadiance = cwiseMax(radiance);
    }
    
    __device__ Device::QuadLight::QuadLight()
    {
        Prepare();
    }

    __device__ void Device::QuadLight::Prepare()
    {        
        m_emitterArea = m_params.light.transform.scale().x * m_params.light.transform.scale().y;
        m_boundingRadius = length(m_params.light.transform.scale() * 0.5f);
        m_lightNormal = normalize(m_params.light.transform.fwd[2]);
    }

    __device__ float Device::QuadLight::Estimate(const Ray& incident, const HitCtx& hitCtx) const
    {  
        const vec3 originDir = m_params.light.transform.trans() - hitCtx.hit.p;
        const float originDist2 = length2(originDir);

        // Shading point is inside the bounding radius, so cap it
        if (originDist2 <= sqr(m_boundingRadius))
        {
            return m_params.peakRadiantIntensity / sqr(m_boundingRadius);
        }   

        // Fast approx estimate of irradiance
        float peakIrradiance = m_params.peakRadiantIntensity / originDist2;

        // Slower but more accurate estimate of peak irradiance
        //const float peakIrradiance = m_params.peakRadiance * kTwoPi * (1 - sqrt(originDist2 - sqr(m_discRadius)) / sqrt(originDist2));

        // Light is too far away to make a contribution. 
        // FIXME: This needs to be corrected in the event that we start using non-Lambertian BRDFs
        if (peakIrradiance < kPeakIntensityMinThreshold) { return 0.0f; }

        // If the light is rotated away from the shading normal, we're done
        if (dot(m_params.light.transform.fwd[2], originDir) >= 0.0f) { return 0.0f; }

        // If the entire bounding sphere of the light is below the horizon, exclude the light.
        // TODO: This is an approximation of the real visibility function, but it works okay. Find a better function later. 
        float cosTheta = dot(originDir, hitCtx.hit.n);
        if (cosTheta < 0.0f && sqr(cosTheta) > sqr(m_boundingRadius)) { return 0.0f; }

        return peakIrradiance * max(0.0f, -dot(originDir / sqrt(originDist2), m_lightNormal));
    }
    
    __device__ bool Device::QuadLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec2 xi, vec3& extant, vec3& L, float& pdfLight) const
    {
        // Sample a point on the light 
        const vec3& hitPos = hitCtx.hit.p;
        const vec3& normal = hitCtx.hit.n;

        xi -= vec2(0.5f);
        const vec3 lightPos = m_params.light.transform.PointToWorldSpace(vec3(xi, 0.0f));

        // Compute the normalised extant direction based on the light position local to the shading point
        extant = lightPos - hitPos;
        float lightDist = length(extant);
        extant /= lightDist;

        // Test if the emitter is behind the shading point
        if (dot(extant, normal) <= 0.0f) { return false; }
      
        vec3 lightNormal = m_params.light.transform.PointToWorldSpace(vec3(xi, 1.0f)) - lightPos;
        float cosPhi = dot(extant, normalize(lightNormal));
        
        // Test if the emitter is rotated away from the shading point
        if (cosPhi > 0.0f) { return false; }

        // Compute the projected solid angle of the light        
        float solidAngle = -cosPhi * m_emitterArea / max(1e-10f, sqr(lightDist));

        // Compute the PDFs of the light
        pdfLight = 1.0f / solidAngle;

        // Calculate the ray throughput in the event that is hits the light
        L = m_params.radiance * solidAngle;
        return true;
    }

    __device__ bool Device::QuadLight::Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const
    {        
        const float cosPhi = dot(hitCtx.hit.n, incident.od.d);
        if (cosPhi > 0.0f)
        {
            return false;
        }
        
        const float solidAngle = -cosPhi * m_emitterArea / sqr(incident.tNear);
        pdfLight = 1 / solidAngle;
        L = m_params.radiance;
        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::QuadLight::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLight) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::QuadLight>(id, json);
    }
    
    __host__  Host::QuadLight::QuadLight(const std::string& id, const ::Json::Node& jsonNode) :
        Host::Light(id, jsonNode),
        cu_deviceData(nullptr)
    {
        // Instantiate the emitter material
        m_lightMaterialAsset = CreateAsset<Host::EmitterMaterial>(id + "_material", kRenderObjectIsChild);
        Assert(m_lightMaterialAsset);

        // Instantiate the plane tracable
        m_lightPlaneAsset = CreateAsset<Host::Plane>(id + "_planeTracable", kRenderObjectIsChild);
        Assert(m_lightPlaneAsset);
        m_lightPlaneAsset->SetBoundMaterialID(id + "_material");

        // Finally, instantitate the light itself 
        cu_deviceData = InstantiateOnDevice<Device::QuadLight>(id);
        FromJson(jsonNode, ::Json::kRequiredWarn);
    }

    __host__ void Host::QuadLight::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __host__ void Host::QuadLight::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Light::FromJson(node, flags);      

        // Pull the parameters from the JSON dictionary and create a transform
        m_params.FromJson(node, flags);
        m_params.light.transform.FromJson(node, flags);
       
        // Update the attributes of the child objects
        m_lightPlaneAsset->UpdateParams(m_params.light.transform, true);
        m_lightMaterialAsset->UpdateParams(m_params.radiance);
     
        // Synchronise with the decvice
        SynchroniseObjects(cu_deviceData, m_params);
    }

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::QuadLight::GetChildObjectHandles()
    {
        return std::vector<AssetHandle<Host::RenderObject>>(
            {
            m_lightPlaneAsset.StaticCast<Host::RenderObject>(),
            m_lightMaterialAsset.StaticCast<Host::RenderObject>() 
            }
         );
    }

    __host__ AssetHandle<Host::Tracable> Host::QuadLight::GetTracableHandle()
    {
        AssertMsgFmt(m_lightPlaneAsset, "QuadLight object '%s' was not properly initialised.", GetAssetID().c_str());

        return m_lightPlaneAsset.StaticCast<Host::Tracable>();
    }
}