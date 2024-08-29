#pragma once

#include "core/3d/primitives/GenericIntersector.cuh"
#include "core/3d/Ctx.cuh"
#include "core/3d/Transform.cuh"
#include "Integrator.cuh"
#include "core/containers/Vector.cuh"
#include "TextureShader.cuh"
#include "../scene/textures/TextureMap.cuh"

namespace Enso
{
    namespace Device
    {
        // Ray-plane intersection test
        __device__  bool RayPlane(Ray& ray, HitCtx& hit, const BidirectionalTransform& transform, const bool isBounded)
        {
            const RayBasic localRay = transform.RayToObjectSpace(ray.od);
            const float t = Intersector::RayPlane(localRay);
            if (t <= 0.0 || t >= ray.tNear)
            {
                return false;
            }
            else
            {
                const float u = (localRay.o.x + localRay.d.x * t) + 0.5f;
                const float v = (localRay.o.y + localRay.d.y * t) + 0.5f;

                if (isBounded && (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f)) { return false; }

                ray.tNear = t;
                ray.SetFlag(kRayBackfacing, localRay.o.z < 0.0f);
                hit.n = transform.NormalToWorldSpace(vec3(0.0, 0.0, 1.0));
                hit.uv = vec2(u, v);

                return true;
            }
        }

        // Ray-sphere intersection test
        __device__ bool RaySphere(Ray& ray, HitCtx& hit, const BidirectionalTransform& transform)
        {
            const RayBasic localRay = transform.RayToObjectSpace(ray.od);

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
                hit.n = transform.NormalToWorldSpace(n);
                ray.SetFlag(kRayBackfacing, dot(localRay.o, localRay.o) < 1.0);

                return true;
            }
        }

        __device__ bool RayBox(Ray& ray, HitCtx& hit, const BidirectionalTransform& transform, const vec3& size)
        {
            const RayBasic localRay = transform.RayToObjectSpace(ray.od);

            vec3 tNearPlane, tFarPlane;
            for (int dim = 0; dim < 3; dim++)
            {
                if (fabsf(localRay.d[dim]) > 1e-10)
                {
                    float t0 = (size[dim] * 0.5f - localRay.o[dim]) / localRay.d[dim];
                    float t1 = (-size[dim] * 0.5f - localRay.o[dim]) / localRay.d[dim];
                    if (t0 < t1) { tNearPlane[dim] = t0;  tFarPlane[dim] = t1; }
                    else { tNearPlane[dim] = t1;  tFarPlane[dim] = t0; }
                }
            }

            float tNearMax = cwiseMax(tNearPlane);
            float tFarMin = cwiseMin(tFarPlane);
            if (tNearMax > tFarMin) { return false; }  // Ray didn't hit the box

            float tNear;
            if (tNearMax > 0.0) { tNear = tNearMax; }
            else if (tFarMin > 0.0) { tNear = tFarMin; }
            else { return false; } // Box is behind the ray

            if (tNear > ray.tNear) { return false; }

            vec3 hitLocal = localRay.o + localRay.d * tNear;
            int normPlane = (fabsf(hitLocal.x / size.x) > fabsf(hitLocal.y / size.y)) ?
                            ((fabsf(hitLocal.x / size.x) > fabsf(hitLocal.z / size.z)) ? 0 : 2) :
                            ((fabsf(hitLocal.y / size.y) > fabsf(hitLocal.z / size.z)) ? 1 : 2);

            vec3 n = kZero;
            n[normPlane] = sign(hitLocal[normPlane]);

            ray.tNear = fmaxf(0.0f, tNear);
            hit.n = transform.NormalToWorldSpace(n);
            ray.SetFlag(kRayBackfacing, cwiseMax(abs(localRay.o)) < size[normPlane]);

            return true;
        }   

        // Ray-cylinder intersection test
        __device__ bool RayCylinder(Ray& ray, HitCtx& hit, const BidirectionalTransform& transform, float height)
        {
            const RayBasic localRay = transform.RayToObjectSpace(ray.od);

            // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
            float a = dot(localRay.d.xy, localRay.d.xy);
            float b = 2.0 * dot(localRay.d.xy, localRay.o.xy);
            float c = dot(localRay.o.xy, localRay.o.xy) - 1.0;

            // Intersect the unbounded cylinder
            vec2 tNearCyl, tFarCyl;
            float b2ac4 = b * b - 4.0f * a * c;
            if (b2ac4 < 0.0) { return false; }
            b2ac4 = sqrt(b2ac4);
            tNearCyl = (-b + b2ac4) / (2.0f * a);
            tFarCyl = (-b - b2ac4) / (2.0f * a);
            sort(tNearCyl.x, tFarCyl.x);

            // Intersect the caps
            tNearCyl.y = (-height * 0.5 - localRay.o.z) / localRay.d.z;
            tFarCyl.y = (height * 0.5 - localRay.o.z) / localRay.d.z;
            sort(tNearCyl.y, tFarCyl.y);

            float tNearMax = fmaxf(tNearCyl.x, tNearCyl.y);
            float tFarMin = fminf(tFarCyl.x, tFarCyl.y);
            if (tNearMax > tFarMin) { return false; }  // Ray didn't hit 

            float tNear;
            if (tNearMax > 0.0 && tNearMax < ray.tNear) { tNear = tNearMax; }
            else if (tFarMin > 0.0 && tFarMin < ray.tNear) { tNear = tFarMin; }
            else { return false; } // Box is behind the ray

            const vec3 i = localRay.o + localRay.d * tNear;

            ray.tNear = tNear;
            hit.n = transform.NormalToWorldSpace((tNearCyl.x < tNearCyl.y) ? vec3(0.0, 0.0, sign(i.z)) : vec3(i.xy, 0.));
            ray.SetFlag(kRayBackfacing, dot(localRay.o, localRay.o) < 1.0);

            return true;
        }

        __device__ int TraceGeo(Ray& ray, HitCtx& hit, const Device::Vector<BidirectionalTransform>& transforms, const Device::Vector<Device::Texture2D*>& textures)
        {
            hit.matID = kMatInvalid;

            const int kNumSpheres = transforms.size() - 2;
            for (int primIdx = 0; primIdx < kNumSpheres; ++primIdx)
            {
                bool isHit = false;
                switch (primIdx % 3)
                {
                case 0:
                    isHit = RayCylinder(ray, hit, transforms[primIdx], 1.); break;
                case 1:
                    isHit = RaySphere(ray, hit, transforms[primIdx]); break;
                case 2:
                    isHit = RayBox(ray, hit, transforms[primIdx], kOne * 0.7); break;
                }
                
                if (isHit)
                {
                    //hit.matID = kMatLambertian;                    
                    switch (primIdx)
                    {
                    case 0:
                        hit.matID = kMatPerfectSpecular; break;
                    case 1:
                        hit.matID = kMatPerfectDielectric;
                        hit.alpha = 1.5;
                        break;
                    case 2:
                        hit.matID = kMatLambertian;
                        break;
                    case 3:
                        hit.matID = kMatCompound;
                        hit.alpha = 0.2;
                        break;
                    default:
                        hit.matID = kMatRoughSpecular;
                        hit.alpha = mix(0.01, 0.5, sqr(float(primIdx - 4) / float(kNumSpheres - 4)));
                        break;
                    }

                    hit.albedo = mix(kOne, Hue(float(primIdx) / float(kNumSpheres)), 0.7f);
                }
            }
           
            if (RayPlane(ray, hit, transforms[kNumSpheres+1], true))
            {
                hit.matID = kMatEmitter;
                hit.albedo = kOne;
            }

            if (RayPlane(ray, hit, transforms[kNumSpheres], false))
            {
                hit.matID = kMatRoughSpecular;
                //hit.matID = kMatLambertian;

                //float tex = Texture::EvaluateFBM2D(hit.uv, 20.f, 1);
                //tex = 1.f / (1. + expf(mix(-1.f, 1.f, fbm) * 3.f));
                //float tex = Texture::EvaluateGrid2D(hit.uv * 10., 0.1);
                const float tex = textures[0]->Evaluate(hit).x;
                hit.alpha = mix(0.01, 0.5, (tex));
                hit.albedo = kOne * 1.f;
            }

            // Make all materials diffuse Lambert
            //if (hit.matID != kMatInvalid && hit.matID != kMatEmitter) { hit.matID = kMatLambertian; }

            return hit.matID;
        }
    }
}