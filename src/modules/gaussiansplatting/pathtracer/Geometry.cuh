#pragma once

#include "core/3d/primitives/GenericIntersector.cuh"
#include "core/3d/Ctx.cuh"
#include "core/3d/Transform.cuh"
#include "Integrator.cuh"
#include "core/Vector.cuh"
#include "TextureShader.cuh"

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

        __device__ int Trace(Ray& ray, HitCtx& hit, const Device::Vector<BidirectionalTransform>& transforms)
        {
            hit.matID = kMatInvalid;

            const int kNumSpheres = transforms.Size() - 2;
            for (int sphereIdx = 0; sphereIdx < kNumSpheres; ++sphereIdx)
            {
                if (RaySphere(ray, hit, transforms[sphereIdx]))
                {
                    switch (sphereIdx)
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
                        hit.alpha = mix(0.01, 0.5, sqr(float(sphereIdx - 4) / float(kNumSpheres - 4)));
                        break;
                    }
                }
            }

            if (RayPlane(ray, hit, transforms[kNumSpheres], true))
            {
                hit.matID = kMatRoughSpecular;

                float fbm = FBM2D(hit.uv, 20.f, 5);
                fbm = 1.f / (1. + expf(mix(-1.f, 1.f, fbm) * 3.f));
                hit.alpha = mix(0.01, 0.5, sqr(fbm));
                ray.weight *= 0.7;
            }
            if (RayPlane(ray, hit, transforms[kNumSpheres+1], true))
            {
                hit.matID = kMatEmitter;
            }

            // Make all materials diffuse Lambert
            //if(hit.matID != kInvalidMaterial && hit.matID != kMatEmitter)  hit.matID = kMatLambertian;

            return hit.matID;
        }
    }
}