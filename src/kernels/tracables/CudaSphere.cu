#include "CudaSphere.cuh"

namespace Cuda
{
    __device__  bool Device::Sphere::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        RayBasic localRay = m_transform.RayToObjectSpace(ray.od);

        // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
        float a = dot(localRay.d, localRay.d);
        float b = 2.0 * dot(localRay.d, localRay.o);
        float c = dot(localRay.o, localRay.o) - 1.0;

        float t0, t1;
        if (!quadraticSolve(a, b, c, t0, t1)) { return false; }

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

        hit = m_transform.HitToWorldSpace(hit);

        ray.tNear = tNear;
        hitCtx.Set(hit, false, vec2(0.0f), 1e-5f);

        return true;
    }

    __host__  Host::Sphere::Sphere()
        : cu_deviceData(nullptr)
    {
        m_hostData.m_transform.MakeIdentity();

        InstantiateOnDevice(&cu_deviceData, m_hostData.m_transform);
    }

    __host__ void Host::Sphere::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }
}