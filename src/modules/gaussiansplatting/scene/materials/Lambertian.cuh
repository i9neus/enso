#pragma once

#include "Material.cuh"
#include "core/math/Mappings.cuh"
#include "core/3d/Basis.cuh"

namespace Enso
{
    namespace Device
    {
        class Lambertian : public Device::Material
        {
        public:
            __device__ virtual float Sample(const vec2& xi, const vec3& i, const vec3& n, vec3& o, float& weight) const override final
            {
                // Sample the Lambertian direction
                vec3 r = vec3(SampleUnitDisc(xi), 0.0f);
                r.z = sqrt(1.0 - sqr(r.x) - sqr(r.y));

                // Transform it to world space
                o = CreateBasis(n) * r;
                return r.z / kPi;
            }

            __device__ virtual float Evaluate(const vec3& i, const vec3& o, const vec3& n) const override final
            {
                return 1 / kPi;
            }

            __device__ virtual bool     IsPerfectSpecular() const { return false; }

        };
    }

    namespace Host
    {
        class Lambertian : public Host::Material
        {
        public:
            __host__ Lambertian(const Asset::InitCtx& initCtx, const int albedoIdx) :
                Material(initCtx, albedoIdx),
                cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::Lambertian>(*this))
            {
                Material::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Material>(cu_deviceInstance)); 
                Synchronise(kSyncParams);
            }

            __host__ virtual ~Lambertian() noexcept
            {
                AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
            }

        protected:
            Device::Lambertian* cu_deviceInstance = nullptr;
        };
    }
}