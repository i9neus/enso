#pragma once

#include "core/3d/Ctx.cuh"
#include "core/3d/Ray.cuh"
#include "core/3d/bxdfs/BxDF.cuh"
#include "core/3d/bxdfs/Lambert.cuh"
#include "core/3d/bxdfs/GGX.cuh"
#include "core/3d/bxdfs/Specular.cuh"
#include "core/math/samplers/Dither.cuh"
#include "Scene.cuh"
#include "QuadLight.cuh"

namespace Enso
{
    enum MaterialType : int
    {
        kMatInvalid = -1,
        kMatEmitter = 0,
        kMatRoughSpecular = 1,
        kMatRoughDielectric = 2,
        kMatPerfectSpecular = 3,
        kMatPerfectDielectric = 4,
        kMatLambertian = 5,
        kMatCompound = 6
    };

#define kGenerateNothing 0
#define kGeneratedDirect 1
#define kGeneratedIndirect 2

#define kModePathTraced 0
#define kModeNEE 1

    __device__ __forceinline__ float PowerHeuristic(float pdf1, float pdf2)
    {
        return saturatef(sqr(pdf1) / fmaxf(1e-10, sqr(pdf1) + sqr(pdf2)));
    }

    __device__ float SampleEmitter(const Ray& incident, Ray& extant, const HitCtx& hit, const BidirectionalTransform& emitterTrans, const vec2& xi)
    {
        vec3 i = -incident.od.d;
        float emitterPdf = SampleQuadLight(incident, extant, hit, emitterTrans, xi);
        if (emitterPdf <= 0.) { return 0.; }

        float bxdfPdf;
        switch (hit.matID)
        {
        case kMatLambertian:
        {
            bxdfPdf = BxDF::EvaluateLambertian();
            break;
        }
        case kMatRoughSpecular:
        {
            bxdfPdf = BxDF::EvaluateMicrofacetReflectorGGX(i, extant.od.d, hit.n, hit.alpha);
            break;
        }
        default:
            return 0.; // Should never be here!
        }

        // Lambert cosine factor
        bxdfPdf *= dot(extant.od.d, hit.n);

        // Apply power heuristic up-weighted by a factor to two to account for stochastic branching
        extant.weight *= 2. * bxdfPdf * PowerHeuristic(emitterPdf, bxdfPdf);

        return emitterPdf;
    }

    __device__ float SampleBxDF(const Ray& incident, Ray& extant, const HitCtx& hit, const BidirectionalTransform& emitterTrans, const vec2& xi, const bool useMis)
    {
        float bxdfPdf = 0.f; 
        float brdfWeight = 1.f;
        vec3 o;
        vec3 kickoff = hit.n * 1e-4;

        // Sample the BxDF
        switch (hit.matID)
        {
            case kMatLambertian:
            {
                bxdfPdf = BxDF::SampleLambertian(xi, hit.n, o);
                break;
            }
            case kMatPerfectSpecular:
            {
                bxdfPdf = BxDF::SamplePerfectSpecular(-incident.od.d, hit.n, o);
                break;
            }
            case kMatPerfectDielectric:
            {
                bxdfPdf = BxDF::SamplePerfectDielectric(xi.y, -incident.od.d, hit.n, hit.alpha, o, kickoff);
                break;
            }
            case kMatRoughSpecular:
            {
                bxdfPdf = BxDF::SampleMicrofacetReflectorGGX(xi, -incident.od.d, hit.n, hit.alpha, o, brdfWeight);
                break;
            }
        }

        // Create the ray
        extant.Construct(incident.Point(), o, kickoff, incident.weight * brdfWeight, incident.depth + 1, incident.InheritedFlags());

        // If this isn't a perfect specular BxDF, flag the ray as scattered
        if (hit.matID != kMatPerfectSpecular && hit.matID != kMatPerfectDielectric) { extant.flags |= kRayScattered; }

        // If this is a light samaple, compute the PDF of the emitter and apply the power heuristic
        if (useMis)
        {
            const float emitterPdf = EvaluateQuadLight(extant, hit, emitterTrans);
            extant.weight *= 2. * PowerHeuristic(bxdfPdf, emitterPdf);
            extant.flags |= kRayDirectSampleBxDF;
        }

        return bxdfPdf;
    }

    __device__ void ShadeDirectSample(const Ray& ray, const HitCtx& hit, vec3& L)
    {
        if (hit.matID == kMatEmitter && !ray.IsBackfacing())
        {
            // If this sample is a light ray, all we need to know is whether or not it hit the light. 
            // If it did, just accumulate the weight which contains the radiant energy from the light sample. 
            if (ray.IsDirectSampleLight())
            {
                L += ray.weight;
            }
            // Handle the light as a perfect blackbody that reflects no energy
            else
            {
                L += kEmitterRadiance * ray.weight;
            }
        }
    }

    __device__ int Shade(const Ray& incidentRay, Ray& indirectRay, Ray& directRay, HitCtx& hit, RenderCtx& renderCtx, const BidirectionalTransform& emitterTrans, int renderMode, vec3& L)
    {
        // Emitters don't reflect light...
        if (hit.matID == kMatEmitter)
        {
            // If this is a specular ray or we're in path tracing mode where direct light isn't explicitly evaluated, add its contribution here
            if (!incidentRay.IsBackfacing() && (!incidentRay.IsScattered() || renderMode == kModePathTraced))
            {
                L += kEmitterRadiance * incidentRay.weight;
            }
            return 0;
        }

        // Generate some random numbers
        //vec4 xi = Rand(renderCtx.rng);
        vec4 xi = renderCtx.Rand(1 + incidentRay.depth);
        float xiSplit = fract(OrderedDither(renderCtx.viewport.xy) + float(renderCtx.frameIdx) / 16.);

        // Compound BxDFs need to be stochastically sampled
        if (hit.matID == kMatCompound)
        {
            hit.matID = (xi.w < xiSplit) ? kMatPerfectSpecular : kMatLambertian;
        }

        if (hit.matID == kMatPerfectSpecular || hit.matID == kMatPerfectDielectric) { renderMode = kModePathTraced; }

        int genFlags = 0;
        // If we're in next-event estimation mode, stochastically sample either the emitter or the BxDF
        if (renderMode == kModeNEE)
        {
            if (xiSplit < 0.5)
            {
                if (SampleBxDF(incidentRay, directRay, hit, emitterTrans, xi.xy, true) > 0)
                {
                    genFlags |= kGeneratedDirect;
                }
            }
            else
            {
                if (SampleEmitter(incidentRay, directRay, hit, emitterTrans, xi.xy) > 0)
                {
                    genFlags |= kGeneratedDirect;
                }
            }
        }

        // Sample the BxDF for the indirect contribution
        if (SampleBxDF(incidentRay, indirectRay, hit, emitterTrans, xi.zw, false) > 0)
        {
            genFlags |= kGeneratedIndirect;
        }

        return genFlags;
    }
}