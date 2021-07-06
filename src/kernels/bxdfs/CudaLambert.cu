#include "CudaLambert.cuh"

#include "generic/JsonUtils.h"

namespace Cuda
{
    __device__ bool Device::LambertBRDF::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, float& pdf) const
    {
        const vec2 xi = renderCtx.Rand<0, 1>();

        // Sample the Lambertian direction
        vec3 r = vec3(SampleUnitDisc(xi), 0.0f);
        r.z = sqrt(1.0 - sqr(r.x) - sqr(r.y));

        pdf = r.z / kPi;
        extant = CreateBasis(hitCtx.hit.n) * r;

        return true;
    }
    __device__ bool Device::LambertBRDF::Evaluate(const vec3& incident, const vec3& extant, const HitCtx& hitCtx, float& weight, float& pdf) const
    {
        weight = dot(extant, hitCtx.hit.n);
        pdf = weight / kPi;
        
        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::LambertBRDF::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kBxDF) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LambertBRDF(json), id);
    }
    
    __host__ Host::LambertBRDF::LambertBRDF(const ::Json::Node& parentNode) :
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::LambertBRDF>();
        Host::BxDF::FromJson(parentNode, ::Json::kRequiredWarn);
    }
    
    __host__ void Host::LambertBRDF::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }
}