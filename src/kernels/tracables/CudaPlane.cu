#include "CudaPlane.cuh"

namespace Cuda
{
    __device__  bool Device::Plane::Intersect(Ray& ray, HitCtx& hitCtx) const
    { 
        Ray::Basic localRay = ray.od.ToObjectSpace(m_matrix);

        // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
        if (fabs(localRay.d.z) < 1e-10) { return false; }

        float t = localRay.o.z / -localRay.d.z;
        if (t <= 0.0 || t >= ray.tNear) { return false; }

        float u = (localRay.o.x + localRay.d.x * t) + 0.5;
        float v = (localRay.o.y + localRay.d.y * t) + 0.5;

        if (m_isBounded && (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)) { return false; }

        // If we've hit the surface and it's the closest intersection, calculate the normal and UV coordinates
        // A more efficient way would be to defer this process to avoid unncessarily computing normals for occuluded surfaces.
        const vec3 hitLocal = localRay.o + localRay.d * t;
        const vec3 hitGlobal = m_invMatrix * hitLocal;
        const vec3 nGlobal = m_invMatrix * (vec3(0.0, 0.0, -1.0) + hitLocal);

        ray.tNear = t;
        hitCtx.Set(normalize(hitGlobal - nGlobal), false, vec2(u, v), 1e-5f);

        return true;
    }

    __host__  Host::Plane::Plane(const bool isBounded)
        : cu_deviceData(nullptr)
    {
        m_hostData.m_matrix = mat4::indentity();
        m_hostData.m_invMatrix = mat4::indentity();
        m_hostData.m_isBounded= isBounded;

        InstantiateOnDevice(&cu_deviceData, m_hostData.m_matrix, m_hostData.m_invMatrix, m_hostData.m_isBounded);
    }

    __host__ void Host::Plane::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }
}