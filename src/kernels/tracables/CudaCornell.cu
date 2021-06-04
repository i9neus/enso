#include "CudaCornell.cuh"

namespace Cuda
{
    __device__  bool Device::Cornell::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        RayBasic localRay = m_transform.RayToObjectSpace(ray.od);

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
        hit.o = localRay.o + localRay.d * t;

        // If we've hit the surface and it's the closest intersection, calculate the normal and UV coordinates
        // A more efficient way would be to defer this process to avoid unncessarily computing normals for occuluded surfaces.
        hit = m_transform.HitToWorldSpace(hit);
        
        if (dot(hit.n, ray.od.o - hit.o) < 0.0f) { hit.n = -hit.n; }

        ray.tNear = t;
        hitCtx.Set(hit, false, uv, 1e-5f);

        return true;
    }

    __host__  Host::Cornell::Cornell()
        : cu_deviceData(nullptr)
    {
        m_hostData.m_transform.MakeIdentity();

        InstantiateOnDevice(&cu_deviceData, m_hostData.m_transform);
    }

    __host__ void Host::Cornell::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }
}