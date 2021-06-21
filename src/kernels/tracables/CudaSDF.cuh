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

        /*template<typename PolyType>
        __device__ __forceinline__ vec4 PolyhedronFace(const vec3& p, const PolyType& poly, const uint& faceIdx, const float& scale)
        {
            const vec3* V = poly.V;
            const vec3& N = poly.N[faceIdx];
            const uchar* F = &poly.F[faceIdx * PolyType::kPolyOrder];
            vec3 grad;

            // Test against each of the polygon's edges
            for (int i = 0; i < PolyType::kPolyOrder; i++)
            {
                const vec3 dv = (V[F[(i + 1) % poly.kPolyOrder]] - V[F[i]]) * scale;
                const vec3 vi = V[F[i]] * scale;
                const vec3 edgeNorm = normalize(cross(dv, N));
                if (dot(edgeNorm, p - vi) > 0.0f)
                {
                    const float t = clamp((dot(p, dv) - dot(vi, dv)) / dot(dv, dv), 0.0f, 1.0f);
                    grad = p - (vi + t * dv);
                    const float gradMag = length(grad);
                    return vec4(gradMag, grad / gradMag);
                }
            }

            // Test against the face itself
            const vec3 v0 = V[F[0]] * scale;
            return (dot(N, p - v0) < 0.0f) ? vec4((dot(p, -N) - dot(v0, -N)), -N) : vec4((dot(p, N) - dot(v0, N)), N);
        }*/
    }
}