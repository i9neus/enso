#pragma once

#include "core/3d/Ctx.cuh"
#include "core/3d/Ray.cuh"
#include "core/3d/bxdfs/BxDF.cuh"
#include "Scene.cuh"
#include "core/3d/Transform.cuh"

namespace Enso
{
    __host__ __device__ float SampleQuadLight(const Ray& incident, Ray& extant, const HitCtx& hit, const BidirectionalTransform& emitterTrans, const vec2& xi)
    {
        // Sample a point on the light 
        vec3 hitPos = incident.Point();

        //vec2 xi = vec2(0.0);
        //uint hash = HashOf(uint(gFragCoord.x), uint(gFragCoord.y));
        //vec2 xi = vec2(HaltonBase2(hash + uint(sampleIdx)), HaltonBase3(hash + uint(sampleIdx))) - 0.5;

        vec3 lightPos = emitterTrans.inv * vec3(xi - 0.5f, 0.f) * emitterTrans.sca + emitterTrans.trans;
        //lightPos = emitterTrans.trans;

        // Compute the normalised extant direction based on the light position local to the shading point
        vec3 outgoing = lightPos - hitPos;
        float lightDist = length(outgoing);
        outgoing /= lightDist;

        // Test if the emitter is behind the shading point
        if (dot(outgoing, hit.n) <= 0.f) { return 0.0f; }

        vec3 lightNormal = normalize(emitterTrans.inv * vec3(0.0f, 0.0f, 1.0f));
        float cosPhi = dot(normalize(hitPos - lightPos), lightNormal);

        // Test if the emitter is rotated away from the shading point
        if (cosPhi < 0.f) { return 0.0f; }

        // Compute the projected solid angle of the light        
        float solidAngle = cosPhi * sqr(emitterTrans.sca) / fmaxf(1e-10f, sqr(lightDist));

        // Create the ray from the sampled BRDF direction
        extant.Construct(hitPos,
            outgoing,
            //(IsBackfacing(ray) ? hit.n : hit.n) * hit.kickoff,
            hit.n * 1e-4f,
            incident.weight * kEmitterRadiance * solidAngle,
            incident.depth + 1,
            kRayDirectSampleLight);

        return 1.0f / fmaxf(1e-10f, solidAngle);
    }

    __host__ __device__  float EvaluateQuadLight(Ray& extant, const HitCtx& hit, const BidirectionalTransform& emitterTrans)
    {
        RayBasic localRay = emitterTrans.RayToObjectSpace(extant.od);
        if (fabsf(localRay.d.z) < 1e-10f) { return 0.0f; }

        float t = localRay.o.z / -localRay.d.z;

        const vec2 uv = (localRay.o.xy + localRay.d.xy * t) + 0.5f;
        if (cwiseMin(uv) < 0.0 || cwiseMax(uv) > 1.0) { return 0.0f; }

        const vec3 lightNormal = normalize(emitterTrans.inv * vec3(0.0f, 0.0f, 1.0f));
        const vec3 lightPos = extant.PointAt(t);

        const float cosPhi = dot(normalize(extant.od.o - lightPos), lightNormal);

        // Test if the emitter is rotated away from the shading point
        if (cosPhi < 0.f) { return 0.0f; }

        float solidAngle = cosPhi * sqr(emitterTrans.sca) / fmaxf(1e-10f, sqr(t));

        //if(!IsVolumetricBxDF(hit))
        {
            const float cosTheta = dot(hit.n, extant.od.d);
            if (cosTheta < 0.0f) { return 0.0f; }

            solidAngle *= cosTheta;
        }

        extant.weight *= kEmitterRadiance;
        return 1.0f / fmaxf(1e-10f, solidAngle);
    }
}