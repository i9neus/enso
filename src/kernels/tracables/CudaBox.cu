#include "CudaBox.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __device__  bool Device::Box::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        if (ray.flags & kRayLightProbe && m_params.renderObject.flags() & kRenderObjectExcludeFromBake) { return false; }

        const RayBasic localRay = RayToObjectSpace(ray.od, m_params.transform);

        vec3 tNearPlane, tFarPlane;
        for (int dim = 0; dim < 3; dim++)
        {
            if (fabs(localRay.d[dim]) > 1e-10f)
            {
                float t0 = (0.5 - localRay.o[dim]) / localRay.d[dim];
                float t1 = (-0.5 - localRay.o[dim]) / localRay.d[dim];
                if (t0 < t1) { tNearPlane[dim] = t0;  tFarPlane[dim] = t1; }
                else { tNearPlane[dim] = t1;  tFarPlane[dim] = t0; }
            }
        }

        const float tNearMax = cwiseMax(tNearPlane);
        const float tFarMin = cwiseMin(tFarPlane);
        if (tNearMax > tFarMin) { return false; }  // Ray didn't hit the box

        float t;
        if (tNearMax > 0.0) { t = tNearMax; }
        else if (tFarMin > 0.0) { t = tFarMin; }
        else { return false; } // Box is behind the ray

        if (t > ray.tNear) { return false; }

        vec3 hitLocal = localRay.PointAt(t);
        int normPlane = (fabs(hitLocal.x) > fabs(hitLocal.y)) ?
                        ((fabs(hitLocal.x) > fabs(hitLocal.z)) ? 0 : 2) :
                        ((fabs(hitLocal.y) > fabs(hitLocal.z)) ? 1 : 2);
        vec3 n = kZero;
        n[normPlane] = sign(hitLocal[normPlane]);

        ray.tNear = t;
        HitPoint hit;
        hit.p = ray.HitPoint();
        hit.n = NormalToWorldSpace(n, m_params.transform);
        if (dot(hit.n, ray.od.o - hit.p) < 0.0f) { hit.n = -hit.n; }

        hitCtx.Set(hit, IsPointInUnitBox(localRay.o), vec2(0.0f), 1e-5f, kNotALight);

        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::Box::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::Box(json), id);
    }

    // Constructor used to instantiate child objects e.g. from quad lights
    __host__  Host::Box::Box()
    {
        AssertMsg(false, "Wrong ctor.");
    }

    // Constructor for user instantiations
    __host__  Host::Box::Box(const ::Json::Node& node)
    {
        cu_deviceData = InstantiateOnDevice<Device::Box>();

        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void Host::Box::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::Box::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Tracable::FromJson(node, flags);

        m_params.FromJson(node, flags);
        RenderObject::SetUserFacingRenderObjectFlags(m_params.renderObject.flags());

        SynchroniseObjects(cu_deviceData, m_params);
    }
}