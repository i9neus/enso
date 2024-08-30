#include "QuadLight.cuh"
#include "../tracables/Primitives.cuh"
#include "core/3d/primitives/GenericIntersector.cuh"

namespace Enso
{    
    __device__ float Device::QuadLight::Sample(const Ray& incident, Ray& extant, const HitCtx& hit, const vec2& xi) const
    {
        // Sample a point on the light 
        vec3 hitPos = incident.Point();

        //vec2 xi = vec2(0.0);
        //uint hash = HashOf(uint(gFragCoord.x), uint(gFragCoord.y));
        //vec2 xi = vec2(HaltonBase2(hash + uint(sampleIdx)), HaltonBase3(hash + uint(sampleIdx))) - 0.5;

        vec3 lightPos = LightSampler::m_params.transform.inv * vec3(xi - 0.5f, 0.f) * LightSampler::m_params.transform.sca + LightSampler::m_params.transform.trans;
        //lightPos = LightSampler::m_params.transform.trans;

        // Compute the normalised extant direction based on the light position local to the shading point
        vec3 outgoing = lightPos - hitPos;
        float lightDist = length(outgoing);
        outgoing /= lightDist;

        // Test if the emitter is behind the shading point
        if (dot(outgoing, hit.n) <= 0.f) { return 0.0f; }

        vec3 lightNormal = normalize(LightSampler::m_params.transform.inv * vec3(0.0f, 0.0f, 1.0f));
        float cosPhi = dot(normalize(hitPos - lightPos), lightNormal);

        // Test if the emitter is rotated away from the shading point
        if (cosPhi < 0.f) { return 0.0f; }

        // Compute the projected solid angle of the light        
        float solidAngle = cosPhi * sqr(LightSampler::m_params.transform.sca) / fmaxf(1e-10f, sqr(lightDist));

        // Create the ray from the sampled BRDF direction
        extant.Construct(hitPos,
            outgoing,
            //(IsBackfacing(ray) ? hit.n : hit.n) * hit.kickoff,
            hit.n * 1e-4f,
            incident.weight * LightSampler::m_params.radiance * solidAngle,
            incident.depth + 1,
            kRayDirectSampleLight);

        return 1.0f / fmaxf(1e-10f, solidAngle);
    }

    __device__  float Device::QuadLight::Evaluate(Ray& extant, const HitCtx& hit) const
    {
        RayBasic localRay = LightSampler::m_params.transform.RayToObjectSpace(extant.od);        
        const float t = Intersector::RayPlane(localRay);
        if (t <= 0.0) { return 0.0f; }

        const vec2 uv = (localRay.o.xy + localRay.d.xy * t) + 0.5f;
        if (cwiseMin(uv) < 0.0f || cwiseMax(uv) > 1.0f) { return 0.0f; }

        const vec3 lightNormal = normalize(LightSampler::m_params.transform.inv * vec3(0.0f, 0.0f, 1.0f));
        const vec3 lightPos = extant.PointAt(t);

        const float cosPhi = dot(normalize(extant.od.o - lightPos), lightNormal);

        // Test if the emitter is rotated away from the shading point
        if (cosPhi < 0.f) { return 0.0f; }

        float solidAngle = cosPhi * sqr(LightSampler::m_params.transform.sca) / fmaxf(1e-10f, sqr(t));

        //if(!IsVolumetricBxDF(hit))
        {
            const float cosTheta = dot(hit.n, extant.od.d);
            if (cosTheta < 0.0f) { return 0.0f; }

            solidAngle *= cosTheta;
        }

        extant.weight *= LightSampler::m_params.radiance;
        return 1.0f / fmaxf(1e-10f, solidAngle);
    }
    
    __host__ Host::QuadLight::QuadLight(const Asset::InitCtx& initCtx, const vec3& radiance, AssetHandle<Host::Tracable>& tracable) :
        LightSampler(initCtx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::QuadLight>(*this))
    {
        LightSampler::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::LightSampler>(cu_deviceInstance));

        m_params.radiance = radiance;
        BindTracable(tracable);

        Synchronise(kSyncParams);
    }

    __host__ bool Host::QuadLight::TryBind(AssetHandle<Host::Tracable>& tracable)
    {
        return tracable.DynamicCast<Host::PlanePrimitive>();      
    }

    __host__ Host::QuadLight::~QuadLight() noexcept
    {
       AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance); 
    }
}