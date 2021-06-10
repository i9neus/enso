#include "CudaPlane.cuh"

namespace Cuda
{
    __device__  bool Device::Plane::Intersect(Ray& ray, HitCtx& hitCtx) const
    { 
        const RayBasic localRay = RayToObjectSpace(ray.od, m_transform);

        // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
        if (fabs(localRay.d.z) < 1e-10) { return false; }

        float t = localRay.o.z / -localRay.d.z;
        if (t <= 0.0 || t >= ray.tNear) { return false; }

        float u = (localRay.o.x + localRay.d.x * t) + 0.5;
        float v = (localRay.o.y + localRay.d.y * t) + 0.5;

        if (m_isBounded && (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)) { return false; }
        
        ray.tNear = t;
        //HitPoint hit = m_transform.HitToWorldSpace(HitPoint(ray.HitPoint(), vec3(0.0f, 0.0f, 1.0f)));
        //if (dot(hit.n, ray.od.o - hit.o) < 0.0f) { hit.n = -hit.n; }

        hitCtx.Set(HitPoint(ray.HitPoint(), NormalToWorldSpace(vec3(0.0f, 0.0f, 1.0f), m_transform)), false, vec2(u, v), 1e-5f);
        return true;
    }

    __host__  Host::Plane::Plane(const BidirectionalTransform& transform, const bool isBounded)
        : cu_deviceData(nullptr)
    {
        m_hostData.m_transform = transform;
        m_hostData.m_isBounded = isBounded;

        cu_deviceData = InstantiateOnDevice<Device::Plane>(m_hostData.m_transform, m_hostData.m_isBounded);
    }

    __host__ void Host::Plane::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }
}