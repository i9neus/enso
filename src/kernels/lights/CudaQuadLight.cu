#include "CudaQuadLight.cuh"

namespace Cuda
{
    __device__ Device::QuadLight::QuadLight(const BidirectionalTransform& transform) : Light(transform)
    {
        m_emitterArea = m_transform.scale.x * m_transform.scale.x;
        m_emitterRadiance = vec3(1.0f);
    }
    
    __device__ bool Device::QuadLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, RayBasic& extant, vec3& L, float& pdfLight) const
    {
        // Sample a point on the light 
        const vec3& hitPos = hitCtx.hit.p;
        const vec3& normal = hitCtx.hit.n;

        const vec2 xi = renderCtx.Rand2() - 0.5f;
        const vec3 lightPos = m_transform.PointToWorldSpace(vec3(xi, 0.0f));

        // Compute the normalised extant direction based on the light position local to the shading point
        vec3 extantDir = lightPos - hitPos;
        float lightDist = length(extantDir);
        extantDir /= lightDist;

        // Test if the emitter is behind the shading point
        if (dot(extantDir, normal) <= 0.0f) { return false; }

        // Test if the emitter is rotated away from the shading point
        vec3 lightNormal = m_transform.PointToWorldSpace(vec3(xi, 1.0f));
        float cosPhi = dot(extantDir, normalize(lightNormal - lightPos));
        if (cosPhi < 0.0f) { return false; }

        // Compute the projected solid angle of the light        
        float solidAngle = cosPhi * min(1e5f, m_emitterArea / sqr(lightDist));

        // Compute the PDFs of the light and BRDF
        float cosTheta = dot(normal, extantDir);
        pdfLight = 1.0f / solidAngle;

        // Calculate the ray throughput in the event that is hits the light
        L = incident.weight * (m_emitterRadiance / m_emitterArea) * solidAngle * cosTheta / kPi;
    }

    __device__ void Device::QuadLight::Evaluate()
    {
    }
    
    __host__  Host::QuadLight::QuadLight()
        : cu_deviceData(nullptr)
    {
        m_hostData.m_transform.MakeIdentity();

        cu_deviceData = InstantiateOnDevice<Device::QuadLight>(m_hostData.m_transform);
    }

    __host__ void Host::QuadLight::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }
}