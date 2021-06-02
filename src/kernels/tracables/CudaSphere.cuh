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

            vec3            m_pos;
            float           m_radius;

        public:
            __device__ Sphere(const mat4& matrix, const mat4& invMatrix, const vec3& pos, const float radius) :
                Tracable(matrix, invMatrix), m_pos(pos), m_radius(radius) {}
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
            __host__ Sphere(const vec3& pos, const float radius);
            __host__ virtual ~Sphere() { OnDestroyAsset(); }
            __host__ virtual void OnDestroyAsset() override final;

            __host__ virtual Device::Sphere* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}