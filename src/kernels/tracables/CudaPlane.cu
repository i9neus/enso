﻿#include "CudaPlane.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ void PlaneParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("bounded", isBounded);
        tracable.ToJson(node);
    }

    __host__ void PlaneParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("bounded", isBounded, flags);
        tracable.FromJson(node, flags);
    }

    __device__  bool Device::Plane::Intersect(Ray& ray, HitCtx& hitCtx) const
    { 
        const RayBasic localRay = RayToObjectSpace(ray.od, m_params.tracable.transform);

        // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
        if (fabs(localRay.d.z) < 1e-10) { return false; }

        float t = localRay.o.z / -localRay.d.z;
        if (t <= 0.0 || t >= ray.tNear) { return false; }

        float u = (localRay.o.x + localRay.d.x * t) + 0.5;
        float v = (localRay.o.y + localRay.d.y * t) + 0.5;

        if (m_params.isBounded && (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)) { return false; }
        
        ray.tNear = t;
        //HitPoint hit = m_transform.HitToWorldSpace(HitPoint(ray.HitPoint(), vec3(0.0f, 0.0f, 1.0f)));
        //if (dot(hit.n, ray.od.o - hit.o) < 0.0f) { hit.n = -hit.n; }

        hitCtx.Set(HitPoint(ray.HitPoint(), NormalToWorldSpace(vec3(0.0f, 0.0f, 1.0f), m_params.tracable.transform)), localRay.o.z < 0.0f, vec2(u, v), 1e-5f);
        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::Plane::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::Plane(json), id);
    }

    // Constructor used to instantiate child objects e.g. from quad lights
    __host__  Host::Plane::Plane(const uint flags)
    {        
        cu_deviceData = InstantiateOnDevice<Device::Plane>();
        RenderObject::SetRenderObjectFlags(flags);
    }

    // Constructor for user instantiations
    __host__  Host::Plane::Plane(const ::Json::Node& node)
    {
        RenderObject::SetRenderObjectFlags(kIsJitterable);
        
        cu_deviceData = InstantiateOnDevice<Device::Plane>();
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void Host::Plane::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::Plane::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Tracable::FromJson(node, flags);        
        SynchroniseObjects(cu_deviceData, PlaneParams(node, flags));
    }

    __host__ void Host::Plane::UpdateParams(const BidirectionalTransform& transform, const bool isBounded)
    {
        m_params = PlaneParams(transform, isBounded);
        SynchroniseObjects(cu_deviceData, m_params);
    }
}