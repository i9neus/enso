#include "CudaSphere.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{     
    __device__  bool Device::Sphere::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        if (ray.flags & kRayLightProbe && m_params.excludeFromBake) { return false; }
        
        const RayBasic localRay = RayToObjectSpace(ray.od, m_params.transform);

        // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
        float a = dot(localRay.d, localRay.d);
        float b = 2.0 * dot(localRay.d, localRay.o);
        float c = dot(localRay.o, localRay.o) - 1.0;

        float t0, t1;
        if (!QuadraticSolve(a, b, c, t0, t1)) { return false; }

        if (t1 < t0)
        {
            float swap = t1;
            t1 = t0;
            t0 = swap;
        }

        float tNear = ray.tNear;
        HitPoint hit;
        if (t0 > 0.0 && t0 < tNear)
        {
            hit.n = localRay.o + localRay.d * t0;
            tNear = t0;
        }
        else if (t1 > 0.0 && t1 < tNear)
        {
            hit.n = localRay.o + localRay.d * t1;
            tNear = t1;
        }
        else { return false; }

        ray.tNear = tNear;
        hit.p = ray.HitPoint();
        hit.n = NormalToWorldSpace(hit.n, m_params.transform);

        hitCtx.Set(hit, length2(localRay.o) < 1.0f, vec2(0.0f), 1e-5f);

        return true;
    }

     __host__ AssetHandle<Host::RenderObject> Host::Sphere::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
     {
         if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

         return AssetHandle<Host::RenderObject>(new Host::Sphere(json), id);
     }

     // Constructor used to instantiate child objects e.g. from sphere lights
     __host__  Host::Sphere::Sphere()
     {
         cu_deviceData = InstantiateOnDevice<Device::Sphere>();
         RenderObject::SetRenderObjectFlags(kIsChildObject);
     }

     // Constructor for user instantiations
    __host__  Host::Sphere::Sphere(const ::Json::Node& node)
    {
        RenderObject::SetRenderObjectFlags(kIsJitterable);
        
        cu_deviceData = InstantiateOnDevice<Device::Sphere>();
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void Host::Sphere::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Tracable::FromJson(node, flags);
        
        SynchroniseObjects(cu_deviceData, TracableParams(node, flags));
    }

    __host__ void Host::Sphere::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::Sphere::UpdateParams(const BidirectionalTransform& transform)
    {
        SynchroniseObjects(cu_deviceData, TracableParams(transform));
    }
}