#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"

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
    }

    namespace Host
    {
        class Tracable : public Host::Asset, public AssetTags<Host::Tracable, Device::Tracable>
        {
        public:
            __host__ virtual Device::Tracable* GetDeviceInstance() const = 0;
        };
    }
}