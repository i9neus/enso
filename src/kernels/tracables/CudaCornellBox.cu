#include "CudaCornellBox.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ void CornellBoxParams::ToJson(::Json::Node& node) const
    {
        tracable.ToJson(node);
        faceMask.ToJson("faceMask", node);
        cameraRayMask.ToJson("cameraRayMask", node);
    }

    __host__ uint CornellBoxParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        tracable.FromJson(node, flags);
        faceMask.FromJson("faceMask", node, flags);
        cameraRayMask.FromJson("cameraRayMask", node, flags);

        return kRenderObjectDirtyAll;
    }

    __device__  bool Device::CornellBox::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        if (ray.flags & kRayLightProbe && m_params.tracable.renderObject.flags() & kRenderObjectExcludeFromBake) { return false; }
        
        const RayBasic localRay = RayToObjectSpace(ray.od, m_params.tracable.transform);

        float t = ray.tNear;
        vec2 uv;
        HitPoint hit;
        for (int face = 0; face < 6; face++)
        {
            if (!(m_params.faceMask() & (1 << face))) { continue; }
            
            int dim = face / 2;
            float side = 2.0f * float(face % 2) - 1.0f;

            if (fabs(localRay.d[dim]) < 1e-10f) { continue; }

            float tFace = (0.5 * side - localRay.o[dim]) / localRay.d[dim];
            if (tFace <= 0.0 || tFace >= t) { continue; }

            int a = (dim + 1) % 3, b = (dim + 2) % 3;
            vec2 uvFace = vec2((localRay.o[a] + localRay.d[a] * tFace) + 0.5f,
                (localRay.o[b] + localRay.d[b] * tFace) + 0.5f);

            if (uvFace.x < 0.0f || uvFace.x > 1.0f || uvFace.y < 0.0f || uvFace.y > 1.0f) { continue; }

            if (face == 5 && ray.depth <= 1 && localRay.o.z > 0.5) { break; }

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

        hitCtx.Set(hit, false, uv, 1e-5f, kNotALight);

        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::CornellBox::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::CornellBox>(id, json);
    }

    // Constructor used to instantiate child objects e.g. from quad lights
    __host__  Host::CornellBox::CornellBox(const std::string& id) : Tracable(id)
    {
        AssertMsg(false, "Wrong ctor.");
    }

    // Constructor for user instantiations
    __host__  Host::CornellBox::CornellBox(const std::string& id, const ::Json::Node& node) : Tracable(id)
    {
        cu_deviceData = InstantiateOnDevice<Device::CornellBox>(id);

        FromJson(node, ::Json::kSilent);
    }

    __host__ void Host::CornellBox::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __host__ uint Host::CornellBox::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Tracable::FromJson(node, flags);

        m_params.FromJson(node, flags);
        RenderObject::SetUserFacingRenderObjectFlags(m_params.tracable.renderObject.flags());

        SynchroniseObjects(cu_deviceData, m_params);

        return kRenderObjectDirtyAll;
    }
}