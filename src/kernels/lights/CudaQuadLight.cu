#include "CudaQuadLight.cuh"
#include "../tracables/CudaPlane.cuh"
#include "../materials/CudaEmitterMaterial.cuh"
#include "../math/CudaColourUtils.cuh"

#include "generic/JsonUtils.h"

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

        radiance = HSVToRGB(light.colourHSV()) * std::pow(2.0f, light.intensity()) / (light.transform.scale().x * light.transform.scale().y);
    }
    
    __device__ Device::QuadLight::QuadLight()
    {
        Prepare();
    }

    __device__ void Device::QuadLight::Prepare()
    {        
        m_emitterArea = m_params.light.transform.scale().x * m_params.light.transform.scale().y;
    }

    __device__ float Device::QuadLight::Estimate(const Ray& incident, const HitCtx& hitCtx) const
    {
        return 1.0f;
    }
    
    __device__ bool Device::QuadLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdfLight) const
    {
        // Sample a point on the light 
        const vec3& hitPos = hitCtx.hit.p;
        const vec3& normal = hitCtx.hit.n;

        const vec2 xi = renderCtx.rng.Rand<2, 3>() - vec2(0.5f);
        const vec3 lightPos = m_params.light.transform.PointToWorldSpace(vec3(xi, 0.0f));

        // Compute the normalised extant direction based on the light position local to the shading point
        extant = lightPos - hitPos;
        float lightDist = length(extant);
        extant /= lightDist;

        // Test if the emitter is behind the shading point
        if (dot(extant, normal) <= 0.0f) { return false; }

        // Test if the emitter is rotated away from the shading point
        vec3 lightNormal = m_params.light.transform.PointToWorldSpace(vec3(xi, 1.0f)) - lightPos;
        float cosPhi = dot(extant, normalize(lightNormal));
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

        return AssetHandle<Host::RenderObject>(new Host::QuadLight(json, id), id);
    }
    
    __host__  Host::QuadLight::QuadLight(const ::Json::Node& jsonNode, const std::string& id)
        : cu_deviceData(nullptr)
    {
        // Instantiate the emitter material
        m_lightMaterialAsset = AssetHandle<Host::EmitterMaterial>(new Host::EmitterMaterial(kIsChildObject), id + "_material");
        Assert(m_lightMaterialAsset);

        // Instantiate the plane tracable
        m_lightPlaneAsset = AssetHandle<Host::Plane>(new Host::Plane(kIsChildObject), id + "_planeTracable");
        Assert(m_lightPlaneAsset);
        m_lightPlaneAsset->SetBoundMaterialID(id + "_material");

        // Finally, instantitate the light itself 
        cu_deviceData = InstantiateOnDevice<Device::QuadLight>();
        FromJson(jsonNode, ::Json::kRequiredWarn);
    }

    __host__ void Host::QuadLight::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
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