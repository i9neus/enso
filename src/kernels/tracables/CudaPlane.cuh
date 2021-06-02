#pragma once

#include "CudaTracable.cuh"

namespace Cuda
{
    namespace Host { class Plane; }

    namespace Device
    {
        class Plane : public Device::Tracable
        {
            friend Host::Plane;
        protected:
            Plane() = default;

            bool m_isBounded;

        public:
            __device__ Plane(const mat4& matrix, const mat4& invMatrix, const bool isBounded) :
                Tracable(matrix, invMatrix), m_isBounded(isBounded) {}
            __device__ ~Plane() = default;

            __device__ bool Intersect(Ray& ray, HitCtx& hit) const;
        };
    }

    namespace Host
    {
        class Plane : public Host::Tracable
        {
        private:
            Device::Plane* cu_deviceData;
            Device::Plane  m_hostData;

        public:
            __host__ Plane(const bool isBounded);
            __host__ virtual ~Plane() { OnDestroyAsset(); }
            __host__ virtual void OnDestroyAsset() override final;

            __host__ virtual Device::Plane* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}