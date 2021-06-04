#pragma once

#include "CudaLight.cuh"

namespace Cuda
{
    namespace Host { class QuadLight;  }

    namespace Device
    {
        class QuadLight : public Device::Light
        {
            friend Host::QuadLight;
        protected:
            QuadLight() = default;

            float m_emitterArea;
            vec3  m_emitterRadiance;

        public:
            __device__ QuadLight(const BidirectionalTransform& transform);
            __device__ ~QuadLight() = default;

            __device__ bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, RayBasic& extant, vec3& L, float& pdf) const;
            __device__ void Evaluate();
        };
    }

    namespace Host
    {
        class QuadLight : public Host::Light
        {
        private:
            Device::QuadLight* cu_deviceData;
            Device::QuadLight  m_hostData;

        public:
            __host__ QuadLight();
            __host__ virtual ~QuadLight() { OnDestroyAsset(); }
            __host__ virtual void OnDestroyAsset() override final;

            __host__ virtual Device::QuadLight* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}