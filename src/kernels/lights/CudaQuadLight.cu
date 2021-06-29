#include "CudaQuadLight.cuh"
#include "../tracables/CudaPlane.cuh"

#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ QuadLightParams::QuadLightParams(const ::Json::Node& node) :
        QuadLightParams()
    { 
         FromJson(node, ::Json::kRequiredWarn); 
    }
    
    __host__ void QuadLightParams::ToJson(::Json::Node& node) const
    {
        node.AddArray("position", std::vector<float>({ position.x, position.y, position.z }));
        node.AddArray("orientation", std::vector<float>({ orientation.x, orientation.y, orientation.z }));
        node.AddArray("scale", std::vector<float>({ scale.x, scale.y, scale.z }));

        node.AddValue("intensity", intensity);
        node.AddArray("colour", std::vector<float>({ colour.x, colour.y, colour.z }));
    }

    __host__ void QuadLightParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetVector("position", position, flags);
        node.GetVector("orientation", orientation, flags);
        node.GetVector("scale", scale, flags);

        node.GetValue("intensity", intensity, flags);
        node.GetVector("colour", colour, flags);
    }
    
    __device__ Device::QuadLight::QuadLight()
    {
        Prepare();
    }

    __device__ void Device::QuadLight::Prepare()
    {        
        m_emitterArea = m_params.transform.scale.x * m_params.transform.scale.x;
        m_emitterRadiance = m_params.colour * math::pow(2.0f, m_params.intensity) / m_emitterArea;
    }
    
    __device__ bool Device::QuadLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdfLight) const
    {
        // Sample a point on the light 
        const vec3& hitPos = hitCtx.hit.p;
        const vec3& normal = hitCtx.hit.n;

        const vec2 xi = renderCtx.Rand<0, 1>() - 0.5f;
        const vec3 lightPos = m_params.transform.PointToWorldSpace(vec3(xi, 0.0f));

        // Compute the normalised extant direction based on the light position local to the shading point
        extant = lightPos - hitPos;
        float lightDist = length(extant);
        extant /= lightDist;

        // Test if the emitter is behind the shading point
        if (dot(extant, normal) <= 0.0f) { return false; }

        // Test if the emitter is rotated away from the shading point
        vec3 lightNormal = m_params.transform.PointToWorldSpace(vec3(xi, 1.0f));
        float cosPhi = dot(extant, normalize(lightNormal - lightPos));
        if (cosPhi < 0.0f) { return false; }

        // Compute the projected solid angle of the light        
        float solidAngle = cosPhi * min(1e5f, m_emitterArea / sqr(lightDist));

        // Compute the PDFs of the light and BRDF
        float cosTheta = dot(normal, extant);
        pdfLight = 1.0f / solidAngle;

        // Calculate the ray throughput in the event that is hits the light
        L = incident.weight * m_emitterRadiance * solidAngle * cosTheta / kPi;
    }

    __device__ void Device::QuadLight::Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const
    {        
        const float solidAngle = dot(hitCtx.hit.n, incident.od.d) * m_emitterArea / sqr(incident.tNear);

        pdfLight = 1 / solidAngle;
        L = m_emitterRadiance;
    }

    __host__ AssetHandle<Host::RenderObject> Host::QuadLight::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::QuadLight(json), id);
    }
    
    __host__  Host::QuadLight::QuadLight(const ::Json::Node& jsonNode)
        : cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::QuadLight>();
        FromJson(jsonNode, ::Json::kRequiredWarn);
    }

    __host__ void Host::QuadLight::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::QuadLight::FromJson(const ::Json::Node& node, const uint flags)
    {
        Json::Node childNode = node.GetChildObject("quadLight", flags);
        if (!childNode) { return; }

        // Pull the parameters from the JSON dictionary and create a transform
        QuadLightParams newParams;
        newParams.FromJson(childNode, flags);
        newParams.transform.FromJson(childNode, flags); 
       
        // Update the transform of the light plane asset so they match
        //m_lightPlaneAsset->SetTransform(newParams.transform);
     
        // Synchronise with the decvice
        SynchroniseObjects(cu_deviceData, newParams);
    }
}