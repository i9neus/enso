#pragma once

#include "CudaMaterial.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class EmitterMaterial;
    }

    namespace Device
    {
        class EmitterMaterial : public Device::Material
        {
            friend Host::EmitterMaterial;

        public:
            __device__ EmitterMaterial() : m_radiance(0.0f) {}
            __device__ ~EmitterMaterial() {}

            vec3 m_radiance;

            __device__ virtual void Evaluate(const HitCtx& hit, vec3& albedo, vec3& incandescence) const override final
            {
                albedo = 0.0f;
                incandescence = m_radiance;
            }
            __device__ void Synchronise(const vec3& radiance) { m_radiance = radiance; }
        };
    }

    namespace Host
    {
        class BxDF;

        class EmitterMaterial : public Host::Material
        {
        private:
            Device::EmitterMaterial* cu_deviceData;

        public:
            __host__ EmitterMaterial();
            __host__ EmitterMaterial(const vec3& radiance);
            __host__ virtual ~EmitterMaterial() = default;

            __host__ virtual void                       OnDestroyAsset() override final;
            __host__ virtual Device::EmitterMaterial*   GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ static std::string                 GetAssetTypeString() { return "emitterMaterial"; }

            __host__ void                               UpdateParams(const vec3& radiance);
        };
    }
}