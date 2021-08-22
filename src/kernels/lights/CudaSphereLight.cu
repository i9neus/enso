#include "CudaSphereLight.cuh"
#include "../tracables/CudaSphere.cuh"
#include "../materials/CudaEmitterMaterial.cuh"

#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ __device__ SphereLightParams::SphereLightParams() :
        position(0.0f),
        orientation(0.0f),
        scale(1.0f),
        intensity(1.0f),
        colour(1.0f),
        radiance(1.0f) {}

    __host__ SphereLightParams::SphereLightParams(const ::Json::Node& node) :
        SphereLightParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void SphereLightParams::ToJson(::Json::Node& node) const
    {
        transform.ToJson(node);

        node.AddValue("intensity", intensity);
        node.AddArray("colour", std::vector<float>({ colour.x, colour.y, colour.z }));
    }

    __host__ void SphereLightParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        transform.FromJson(node, flags);

        node.GetValue("intensity", intensity, flags);
        node.GetVector("colour", colour, flags);

        radiance = colour * std::pow(2.0f, intensity) / (transform.scale().x * transform.scale().y * kFourPi * 0.25f);
    }

    __device__ Device::SphereLight::SphereLight()
    {
        Prepare();
    }

    __device__ void Device::SphereLight::Prepare()
    {
        m_discArea = kPi * sqr(m_params.transform.scale().x);
        m_discRadius = m_params.transform.scale().x;
    }

    __device__ inline mat3 CreateBasisDebug2(vec3 n)
    {
        n = normalize(n);
        vec3 tangent = normalize(cross(n, (fabs(dot(n, vec3(1.0f, 0.0f, 0.0f))) < 0.5f) ? vec3(1.0f, 0.0f, 0.0f) : vec3(0.0f, 1.0f, 0.0f)));
        vec3 cotangent = cross(tangent, n);

        return mat3(tangent, cotangent, n);
    }

    __device__ bool Device::SphereLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdfLight) const
    {        
        vec3 originDir = m_params.transform.trans() - hitCtx.hit.p;
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
        const float projectedDiscRadius = m_discRadius *sqrtf(1.0f - sqr(sinTheta));

        const vec2 disc = SampleUnitDiscLowDistortion(renderCtx.rng.Rand<2, 3>());
        const mat3 basis = CreateBasisDebug2(originDir);
        vec3 sampleDir = basis.x * disc.x * projectedDiscRadius +
                         basis.y * disc.y * projectedDiscRadius +
                         originDir * originDist;
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
        vec3 originDir = m_params.transform.trans() - incident.od.o;
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

        return AssetHandle<Host::RenderObject>(new Host::SphereLight(json, id), id);
    }

    __host__  Host::SphereLight::SphereLight(const ::Json::Node& jsonNode, const std::string& id)
        : cu_deviceData(nullptr)
    {
        // Instantiate the emitter material
        m_lightMaterialAsset = AssetHandle<Host::EmitterMaterial>(new Host::EmitterMaterial(), id + "_material");
        Assert(m_lightMaterialAsset);

        // Instantiate the sphere tracable
        m_lightSphereAsset = AssetHandle<Host::Sphere>(new Host::Sphere(), id + "_planeTracable");
        Assert(m_lightSphereAsset);
        m_lightSphereAsset->SetBoundMaterialID(id + "_material");

        // Finally, instantitate the light itself 
        cu_deviceData = InstantiateOnDevice<Device::SphereLight>();
        FromJson(jsonNode, ::Json::kRequiredWarn);
    }

    __host__ void Host::SphereLight::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::SphereLight::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Light::FromJson(node, flags);

        // Pull the parameters from the JSON dictionary and create a transform
        m_params.FromJson(node, flags);

        // Update the attributes of the child objects
        m_lightSphereAsset->UpdateParams(m_params.transform);
        m_lightMaterialAsset->UpdateParams(m_params.radiance);

        // Synchronise with the decvice
        SynchroniseObjects(cu_deviceData, m_params);
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