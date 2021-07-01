#pragma once

#include "CudaBxDF.cuh"

namespace Json { class Node; }

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

            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, float& pdf) const override final
            {
                const vec2 xi = renderCtx.Rand<0, 1>();
                
                // Sample the Lambertian direction
                vec3 r = vec3(SampleUnitDisc(xi), 0.0f);
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
            __host__ LambertBRDF(const ::Json::Node&);
            __host__ virtual ~LambertBRDF() { OnDestroyAsset(); }

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string&, const AssetType&, const ::Json::Node&);

            __host__ virtual void OnDestroyAsset() override final;
            __host__ static std::string GetAssetTypeString() { return "lambert"; }
            __host__ virtual Device::LambertBRDF* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}