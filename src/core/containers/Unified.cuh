/*
    NOTES:
     - Unified is a small class that uses RAII to manage simple objects that mimic unified memory.
* */


#pragma once

#include "core/assets/AssetAllocator.cuh"

namespace Enso
{
    namespace Host
    {
        template<typename Type, typename = typename std::enable_if_t<std::is_standard_layout<Type>::value> >
        class Unified
        {
        private:
            Type* cu_deviceData;
            Type  m_hostData;

        public:
            __host__ Unified()
            {
                IsOk(cudaMalloc((void**)&cu_deviceData, sizeof(Type)));
            }

            __host__ ~Unified() noexcept
            {
                if (cu_deviceData) { cudaFree(cu_deviceData); }
            } 

            __host__ Unified(const Unified& other)
            {
                IsOk(cudaMalloc((void**)&cu_deviceData, sizeof(Type)));
                IsOk(cudaMemcpy(cu_deviceData, other.cu_deviceData, sizeof(Type)));
                m_hostData = other.m_hostData;
            }

            __host__ Unified(Unified&& other)
            {
                cu_deviceData = other.cu_deviceData;
                m_hostData = other.m_hostData;
                other.cu_deviceData = nullptr;                
            }

            __host__ Type* GetDevicePtr() { return cu_deviceData; }

            __host__ Unified operator =(const Type& rhs)
            {
                if (m_hostData != rhs)
                {
                    m_hostData = rhs;
                    IsOk(cudaMemcpy(cu_deviceData, &m_hostData, sizeof(Type), cudaMemcpyHostToDevice));
                }
                return *this;
            }
            
            __host__ operator Type()
            {
                IsOk(cudaMemcpy(&m_hostData, cu_deviceData, sizeof(Type), cudaMemcpyDeviceToHost));
                return m_hostData;
            }
        };
    }

} // namespace Enso