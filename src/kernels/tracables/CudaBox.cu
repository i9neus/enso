#include "CudaBox.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ void BoxParams::ToJson(::Json::Node& node) const
    {
        tracable.ToJson(node);
    }

    __host__ void BoxParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        tracable.FromJson(node, flags);

        // FIXME: This is a hack because non-linear scaling is broken in the bidirectional transform object. 
        transform.Set(tracable.transform.trans(), tracable.transform.rot(), vec3(1.0f));
        size = tracable.transform.scale() * 0.5f;
    }
    
    __device__  bool Device::Box::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        if (ray.flags & kRayLightProbe && m_params.tracable.renderObject.flags() & kRenderObjectExcludeFromBake) { return false; }

        const RayBasic localRay = RayToObjectSpace(ray.od, m_params.transform);

        vec3 tNearPlane, tFarPlane;
        for (int dim = 0; dim < 3; dim++)
        {
            if (fabs(localRay.d[dim]) > 1e-10f)
            {
                float t0 = (m_params.size[dim] - localRay.o[dim]) / localRay.d[dim];
                float t1 = (-m_params.size[dim] - localRay.o[dim]) / localRay.d[dim];
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
        int normPlane = (fabs(hitLocal.x / m_params.size.x) > fabs(hitLocal.y / m_params.size.y)) ?
                        ((fabs(hitLocal.x / m_params.size.x) > fabs(hitLocal.z / m_params.size.z)) ? 0 : 2) :
                        ((fabs(hitLocal.y / m_params.size.y) > fabs(hitLocal.z / m_params.size.z)) ? 1 : 2);
        vec3 n = kZero;
        n[normPlane] = sign(hitLocal[normPlane]);

        ray.tNear = t;
        HitPoint hit;
        hit.p = ray.HitPoint();
        hit.n = NormalToWorldSpace(n, m_params.transform);
        //if (dot(hit.n, ray.od.o - hit.p) < 0.0f) { hit.n = -hit.n; }

        hitCtx.Set(hit, IsPointInUnitBox(localRay.o / (m_params.size * 2.0f)), vec2(0.0f), 1e-5f, kNotALight);

        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::Box::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::Box>(id, json);
    }

    // Constructor used to instantiate child objects e.g. from quad lights
    __host__  Host::Box::Box(const std::string& id) : 
        Tracable(id)
    {
        AssertMsg(false, "Wrong ctor.");
    }

    // Constructor for user instantiations
    __host__  Host::Box::Box(const std::string& id, const ::Json::Node& node) : 
        Tracable(id)
    {
        cu_deviceData = InstantiateOnDevice<Device::Box>(id);

        FromJson(node, ::Json::kSilent);
    }

    __host__ void Host::Box::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __host__ void Host::Box::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Tracable::FromJson(node, flags);

        m_params.FromJson(node, flags);
        RenderObject::SetUserFacingRenderObjectFlags(m_params.tracable.renderObject.flags());

        SynchroniseObjects(cu_deviceData, m_params);
    }
}