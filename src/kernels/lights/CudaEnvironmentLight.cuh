#pragma once

#include "CudaLight.cuh"

namespace Cuda
{
    namespace Host { class EnvironmentLight; }

    namespace Device
    {
        class EnvironmentLight : public Device::Light
        {
            friend Host::EnvironmentLight;
        protected:
            EnvironmentLight() = default;

            float m_emitterArea;
            vec3  m_emitterRadiance;

        public:
            __device__ EnvironmentLight(const BidirectionalTransform& transform);
            __device__ ~EnvironmentLight() = default;

            __device__ bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, RayBasic& extant, vec3& L, float& pdf) const;
            __device__ void Evaluate();
        };
    }

    namespace Host
    {
        class EnvironmentLight : public Host::Light
        {
        private:
            Device::EnvironmentLight* cu_deviceData;
            Device::EnvironmentLight  m_hostData;

        public:
            __host__ EnvironmentLight();
            __host__ virtual ~EnvironmentLight() { OnDestroyAsset(); }
            __host__ virtual void OnDestroyAsset() override final;

            __host__ virtual Device::EnvironmentLight* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}