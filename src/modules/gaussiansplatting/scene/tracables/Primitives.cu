#include "Primitives.cuh"

#include "core/3d/primitives/GenericIntersector.cuh"
#include "core/3d/Ctx.cuh"
#include "core/3d/Transform.cuh"

namespace Enso
{
    __device__ bool Device::PlanePrimitive::IntersectRay(Ray& ray, HitCtx& hit) const
    {
        const RayBasic localRay = Tracable::m_params.transform.RayToObjectSpace(ray.od);
        const float t = Intersector::RayPlane(localRay);
        if (t <= 0.0 || t >= ray.tNear)
        {
            return false;
        }
        else
        {
            const vec2 uv = (localRay.o.xy + localRay.d.xy * t) + 0.5f;
            if (cwiseMin(uv) < 0.0 || cwiseMax(uv) > 1.0) { return 0.0f; }

            ray.tNear = t;
            ray.SetFlag(kRayBackfacing, localRay.o.z < 0.0f);
            hit.n = Tracable::m_params.transform.NormalToWorldSpace(vec3(0.0, 0.0, 1.0));
            hit.uv = uv;

            return true;
        }
    }

    __host__ Host::PlanePrimitive::PlanePrimitive(const InitCtx& initCtx) :
        Tracable(initCtx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::PlanePrimitive>(*this))
    {
        Tracable::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Tracable>(cu_deviceInstance));
    }

    __host__ Host::PlanePrimitive::~PlanePrimitive()
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __device__ bool Device::SpherePrimitive::IntersectRay(Ray& ray, HitCtx& hit) const
    {
        const RayBasic localRay = Tracable::m_params.transform.RayToObjectSpace(ray.od);

        vec2 t;
        if (!Intersector::RayUnitSphere(localRay, t))
        {
            return false;
        }
        else
        {
            if (t.y < t.x) { swap(t.x, t.y); }

            vec3 n;
            float tNear = ray.tNear;
            if (t.x > 0.0 && t.x < tNear)
            {
                n = localRay.PointAt(t.x);
                tNear = t.x;
            }
            else if (t.y > 0.0 && t.y < tNear)
            {
                n = localRay.PointAt(t.y);
                tNear = t.y;
            }
            else { return false; }

            ray.tNear = tNear;
            hit.n = Tracable::m_params.transform.NormalToWorldSpace(n);
            ray.SetFlag(kRayBackfacing, dot(localRay.o, localRay.o) < 1.0);

            return true;
        }
    }

    __host__ Host::SpherePrimitive::SpherePrimitive(const InitCtx& initCtx) :
        Tracable(initCtx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::SpherePrimitive>(*this))
    {
        Tracable::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Tracable>(cu_deviceInstance));
    }

    __host__ Host::SpherePrimitive::~SpherePrimitive()
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }
}