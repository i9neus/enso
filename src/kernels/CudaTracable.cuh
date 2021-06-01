#pragma once

#include "CudaRay.cuh"
#include "CudaCtx.cuh"

namespace Cuda
{
    namespace Host 
    {  
        class Tracable;
        class Sphere;
    }
    
    namespace Device
    {
        class Tracable : public Device::Asset, public AssetTags<Host::Tracable, Device::Tracable>
        {
        public:
            Tracable() = default;

        protected:
            __device__ Tracable(const mat4& matrix, const mat4& invMatrix) : m_matrix(matrix), m_invMatrix(invMatrix) {}
            __device__ ~Tracable() = default;

            mat4            m_matrix; 
            mat4            m_invMatrix;
        };

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
        class Tracable : public Host::Asset, public AssetTags<Host::Tracable, Device::Tracable>
        {
        public:
            __host__ virtual Device::Tracable* GetDeviceInstance() const = 0;
        };
        
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