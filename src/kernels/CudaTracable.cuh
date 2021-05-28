#pragma once

#include "CudaRay.cuh"

namespace Cuda
{
    namespace Device
    {
        class Tracable
        {
        public:
            Tracable() = default;

            __device__ virtual bool Intersect(const Ray& ray) { return false;  }
        };

        class Sphere : public Tracable
        {
        private:
            vec3            m_pos;
            float           m_radius;

        public:
            Sphere() = default;

            __device__ virtual bool Intersect(const Ray& ray) override final 
            {
                return false;
            }
        };
    }

    namespace Host
    {
        class Sphere : public Device::Sphere, public AssetBase
        {
        private:
            Device::Sphere* cu_deviceSphere; 

        public:
            Sphere() : cu_deviceSphere(nullptr)
            {                
                SafeCreateDeviceInstance(&cu_deviceSphere, static_cast<Device::Sphere*>(this));
            }

            __host__ virtual void OnDestroyAsset() override final
            {
                SafeFreeDeviceMemory(&cu_deviceSphere);
            }
        };
    }
}