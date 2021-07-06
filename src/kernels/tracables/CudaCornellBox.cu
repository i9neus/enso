﻿#include "CudaCornellBox.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ void CornellBoxParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("bounded", isBounded);
        tracable.ToJson(node);
    }

    __host__ void CornellBoxParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("bounded", isBounded, flags);
        tracable.FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ bool CornellBoxParams::operator==(const CornellBoxParams& rhs) const
    {
        return isBounded == rhs.isBounded &&
            tracable == rhs.tracable;
    }

    __device__  bool Device::CornellBox::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        const RayBasic localRay = RayToObjectSpace(ray.od, m_params.tracable.transform);

        float t = ray.tNear;
        vec2 uv;
        HitPoint hit;
        for (int face = 0; face < 5; face++)
        {
            int dim = face / 2;
            float side = 2.0f * float(face % 2) - 1.0f;

            if (fabs(localRay.d[dim]) < 1e-10f) { continue; }

            float tFace = (0.5 * side - localRay.o[dim]) / localRay.d[dim];
            if (tFace <= 0.0 || tFace >= t) { continue; }

            int a = (dim + 1) % 3, b = (dim + 2) % 3;
            vec2 uvFace = vec2((localRay.o[a] + localRay.d[a] * tFace) + 0.5f,
                (localRay.o[b] + localRay.d[b] * tFace) + 0.5f);

            if (uvFace.x < 0.0f || uvFace.x > 1.0f || uvFace.y < 0.0f || uvFace.y > 1.0f) { continue; }

            t = tFace;
            hit.n = kZero;
            uv = uvFace + vec2(1.0f, 0.0f) * float(face);
            hit.n[dim] = side;
        }

        if (t == ray.tNear) { return false; }
        ray.tNear = t;
        hit.p = ray.HitPoint();
        hit.n = NormalToWorldSpace(hit.n, m_params.tracable.transform);
        if (dot(hit.n, ray.od.o - hit.p) < 0.0f) { hit.n = -hit.n; }

        hitCtx.Set(hit, false, uv, 1e-5f);

        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::CornellBox::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::CornellBox(json), id);
    }

    // Constructor used to instantiate child objects e.g. from quad lights
    __host__  Host::CornellBox::CornellBox()
    {
        cu_deviceData = InstantiateOnDevice<Device::CornellBox>();
        RenderObject::MakeChildObject();
    }

    // Constructor for user instantiations
    __host__  Host::CornellBox::CornellBox(const ::Json::Node& node)
    {
        cu_deviceData = InstantiateOnDevice<Device::CornellBox>();
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void Host::CornellBox::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::CornellBox::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Tracable::FromJson(node, flags);

        SynchroniseObjects(cu_deviceData, CornellBoxParams(node, flags));
    }

    __host__ void Host::CornellBox::UpdateParams(const BidirectionalTransform& transform, const bool isBounded)
    {
        SynchroniseObjects(cu_deviceData, CornellBoxParams(transform, isBounded));
    }
}