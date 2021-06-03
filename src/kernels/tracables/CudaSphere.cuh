#pragma once

#include "CudaTracable.cuh"

namespace Cuda
{
    namespace Host  {  class Sphere;   }

    namespace Device
    {
        class Sphere : public Device::Tracable
        {
            friend Host::Sphere;
        protected:
            Sphere() = default;

        public:
            __device__ Sphere(const BidirectionalTransform& transform) : Tracable(transform) {}
            __device__ ~Sphere() = default;

            __device__ bool Intersect(Ray& ray, HitCtx& hit) const;
        };
    }

    namespace Host
    {
        class Sphere : public Host::Tracable
        {
        private:
            Device::Sphere* cu_deviceData;
            Device::Sphere  m_hostData;

        public:
            __host__ Sphere();
            __host__ virtual ~Sphere() { OnDestroyAsset(); }
            __host__ virtual void OnDestroyAsset() override final;

            __host__ virtual Device::Sphere* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}