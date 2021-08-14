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

        radiance = colour * std::pow(2.0f, intensity) / (transform.scale.x * transform.scale.y * kFourPi * 0.25f);
    }

    __device__ Device::SphereLight::SphereLight()
    {
        Prepare();
    }

    __device__ void Device::SphereLight::Prepare()
    {
        m_emitterArea = m_params.transform.scale.x * m_params.transform.scale.y;
    }

    __device__ bool Device::SphereLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdfLight) const
    {        
        vec3 originDirection = m_params.transform.trans - hitCtx.hit.p;
        const float discRadius = 1.0f * m_params.transform.scale.x;
        const float distance = length(originDirection);

        if (distance <= discRadius)
        {
            pdfLight = 0.0f;
            return false;
        }

        originDirection /= distance;
        const mat3 basis = CreateBasis(originDirection);

        const float theta = asinf(discRadius / distance);
        pdfLight = 1.0f / (1.0f - cosf(theta));
        const float solidAngle = kTwoPi / pdfLight;

        const vec3 disc = SampleUnitSphere(renderCtx.rng.Rand<0, 1>()); 
        const vec3 samplePoint = basis.x * disc[0] * discRadius +
                                 basis.y * disc[1] * discRadius +
                                 originDirection * distance;

        if (dot(samplePoint, hitCtx.hit.n) <= 0)
        {
            pdfLight = 0;
            return false;
        }

        extant = normalize(samplePoint);
        L = m_params.radiance * solidAngle;
        return true;
    }

    __device__ bool Device::SphereLight::Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const
    {
        const float radius = 1.0f * m_params.transform.scale.x;
        const float distance = length(m_params.transform.trans - hitCtx.hit.p);

        if (distance <= radius)
        {
            pdfLight = 0.0f;
            L = 0.0f;
            return false;
        }

        const float theta = std::asin(radius / distance);
        pdfLight = 1.0f / (1.0f - std::cos(theta));
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
        m_params.transform.FromJson(node, flags);

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