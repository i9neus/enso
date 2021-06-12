﻿#pragma once

#include "CudaBxDF.cuh"

namespace Cuda
{
    namespace Host { class LambertBRDF; }

    namespace Device
    {
        class LambertBRDF : public Device::BxDF
        {
            friend Host::LambertBRDF;
        protected:


        public:
            LambertBRDF() = default;
            ~LambertBRDF() = default;

            __device__ bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, float& pdf) const
            {
                const vec2 xi = renderCtx.Rand2();
                
                // Sample the Lambertian direction
                vec3 r = vec3(SampleUnitDisc(xi.xy), 0.0f);
                r.z = sqrt(1.0 - sqr(r.x) - sqr(r.y));

                pdf = r.z / kPi;
                extant = CreateBasis(hitCtx.hit.n) * r;  

                return true;
            }

            __device__ float Evaluate(const Ray& incident, const HitCtx& hitCtx, const vec3& extant) const
            {
                return dot(extant, hitCtx.hit.n) * kPi;
            }
        };
    }

    namespace Host
    {
        class LambertBRDF : public Host::BxDF
        {
        private:
            Device::LambertBRDF* cu_deviceData;
            Device::LambertBRDF  m_hostData;

        public:
            __host__ LambertBRDF();
            __host__ virtual ~LambertBRDF() { OnDestroyAsset(); }
            __host__ virtual void OnDestroyAsset() override final;

            __host__ virtual Device::LambertBRDF* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}