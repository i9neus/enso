#pragma once

#include "CudaRay.cuh"
#include "CudaCtx.cuh"

namespace Cuda
{
    namespace Device
    {
        class Tracable : public ManagedPair<Device::Tracable>
        {
        public:
            Tracable() = default;

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) = 0;

        protected:
            mat4            m_matrix; 
            mat4            m_invMatrix;
        };

        class Sphere : virtual public Device::Tracable
        {
        protected:
            vec3            m_pos;
            float           m_radius;

        public:
            Sphere() = default;
            virtual ~Sphere() {}

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) override final;
        };
    }

    namespace Host
    {
        class Tracable : virtual public Device::Tracable, public AssetBase
        {
        public:
            __host__ virtual Device::Tracable* GetDeviceInstance() const = 0;
        };
        
        class Sphere : public Host::Tracable, public Device::Sphere
        {
        private:
            Device::Tracable* cu_deviceSphere;

        public:
            Sphere(const vec3& pos, const float radius) : cu_deviceSphere(nullptr)
            {                
                Device::Sphere::m_pos = pos;
                Device::Sphere::m_radius = radius; 
                Device::Tracable::m_matrix = mat4::indentity();
                Device::Tracable::m_invMatrix = mat4::indentity();
                
                SafeCreateDeviceInstance(&cu_deviceSphere, static_cast<Device::Sphere*>(this));
            }
            virtual ~Sphere() {}

            __host__ virtual void OnDestroyAsset() override final
            {
                SafeFreeDeviceMemory(&cu_deviceSphere);
            }

            __host__ virtual Device::Tracable* GetDeviceInstance() const override final { return cu_deviceSphere; }
        };
    }
}