#pragma once

#include "../math/CudaMath.cuh"

namespace Cuda
{    
    namespace SDF
    {
        __device__ __forceinline__ vec4 Capsule(const vec3& p, const vec3& v0, const vec3& v1, const float r)
        {
            const vec3 dv = v1 - v0;

            float t = clamp((dot(p, dv) - dot(v0, dv)) / dot(dv, dv), 0.0f, 1.0f);

            vec3 grad = p - (v0 + t * dv);
            float gradMag = length(grad);
            grad /= gradMag;

            return vec4(gradMag - r, grad);
        }

        __device__ __forceinline__ vec4 Torus(const vec3& p, const float& r1, const float& r2)
        {
            const vec3 pPlane = vec3(p.x, 0.0f, p.z);
            float pPlaneLen = length(pPlane);
            const vec3 pRing = (pPlaneLen < 1e-10) ? vec3(0.0) : (p - (pPlane * r1 / pPlaneLen));

            return vec4(length(pRing) - r2, normalize(pRing));
        }

        __device__ __forceinline__ vec4 Box(const vec3& p, const float& size)
        {
            const float F = cwiseMax(abs(p));
            return vec4(F - size, floor(abs(p + vec3(1e-5) * sign(p)) / F) * sign(p));
        }

        __device__ __forceinline__ vec4 Sphere(const vec3& p, const float& r)
        {
            const float pLen = length(p);
            return vec4(pLen - r, vec3(p / pLen));
        }

        template<typename PolyType>
        __device__ __forceinline__ vec4 PolyhedronFace(const vec3& p, const PolyType& poly, const int faceIdx, const float scale)
        {
            // TODO: Pre-cache the normal
            const vec3* V = poly.V;
            const uchar* F = poly.F;
            const uint j = faceIdx * PolyType::kPolyOrder;

            vec3 n = normalize(cross(V[F[j + 1]] - V[F[j + 0]], V[F[j + 2]] - V[F[j + 0]]));
            vec3 grad;

            // Test against each of the polygon's edges
            for (int i = 0; i < PolyType::kPolyOrder; i++)
            {
                const vec3 dv = (V[F[j + (i + 1) % poly.kPolyOrder]] - V[F[j + i]]) * scale;
                const vec3 vi = V[F[j + i]] * scale;
                const vec3 edgeNorm = normalize(cross(dv, n));
                if (dot(edgeNorm, p - vi) > 0.0f)
                {
                    const float t = clamp((dot(p, dv) - dot(vi, dv)) / dot(dv, dv), 0.0f, 1.0f);
                    grad = p - (vi + t * dv);
                    const float gradMag = length(grad);
                    return vec4(gradMag, grad / gradMag);
                }
            }

            // Test against the face itself
            const vec3 v0 = V[F[j + 0]] * scale;
            if (dot(n, p - v0) < 0.0f) { n = -n; }
            return vec4((dot(p, n) - dot(v0, n)), n);
        }
    }
}