#include "CudaTracable.cuh"

namespace Cuda
{    
    __device__  bool Device::Sphere::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        Ray::Basic localRay = ray.od.ToObjectSpace(m_matrix);

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
        vec3 n;
        if (t0 > 0.0 && t0 < tNear)
        {
            n = localRay.o + localRay.d * t0;
            tNear = t0;
        }
        else if (t1 > 0.0 && t1 < tNear)
        {
            n = localRay.o + localRay.d * t1;
            tNear = t1;
        }
        else { return false; }

        const vec3 hitLocal = n;
        const vec3 nLocal = n * 2;
        const vec3 hitGlobal = m_invMatrix * hitLocal;
        const vec3 nGlobal = m_invMatrix * nLocal;

        ray.tNear = tNear;
        hitCtx.Set(normalize(nGlobal - hitGlobal), dot(localRay.o, localRay.o) < 1.0, vec2(0.0), 1e-5);

        return true;
    }

    __host__  Host::Sphere::Sphere(const vec3& pos, const float radius)
        : cu_deviceData(nullptr)
    {
        m_hostData.m_matrix = mat4::indentity();
        m_hostData.m_invMatrix = mat4::indentity();
        m_hostData.m_pos = pos;
        m_hostData.m_radius = radius;

        InstantiateOnDevice(&cu_deviceData, m_hostData.m_matrix, m_hostData.m_invMatrix, m_hostData.m_pos, m_hostData.m_radius);
    }

    __host__ void Host::Sphere::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }
}