#pragma once

#include "CudaCommonIncludes.cuh"

namespace Cuda
{
    // Tiny wrapper that wraps an object and keeps a 
    template<typename Type, int NumElements = 1>
    class DeviceObjectRAII
    {
    public:
        __host__ DeviceObjectRAII() : cu_deviceObject(nullptr) 
        {
            static_assert(NumElements > 0, "DeviceObjectRAII: NumElements must be > 0");
            static_assert(std::is_standard_layout<Type>::value, "DeviceObjectRAII: Must be standard layout type");
        }

        __host__ ~DeviceObjectRAII()
        {
            IsOk(cudaFree(cu_deviceObject));
        }

        __host__ DeviceObjectRAII(const Type& object) : 
            DeviceObjectRAII()
        {
            Upload(object);
        }

        __host__ DeviceObjectRAII& operator=(const Type& object)
        {   
            Upload(object);
        }

        __host__ void Upload(const Type& object)
        {
            m_hostObject[0] = object;
            Upload();
        }

        __host__ void Upload()
        {
            if (!cu_deviceObject)
            {
                CreateDeviceObject();
            }
            IsOk(cudaMemcpy(cu_deviceObject, &m_hostObject, sizeof(Type) * NumElements, cudaMemcpyHostToDevice));
        }

        __host__ Type& Download()
        {
            if (cu_deviceObject)
            {
                IsOk(cudaMemcpy(&m_hostObject, cu_deviceObject, sizeof(Type) * NumElements, cudaMemcpyDeviceToHost));
            }
            return m_hostObject[0];
        }

        __host__ Type* GetDeviceObject()
        {
            if (!cu_deviceObject)
            {
                CreateDeviceObject();
            }
            return cu_deviceObject;
        }

        __host__ Type* operator->(void) { return &m_hostObject[0]; }
        __host__ const Type* operator->(void) const { return &m_hostObject[0]; }
        __host__ Type& operator[](const int idx) { Assert(idx >= 0 && idx < NumElements); return m_hostObject[idx]; }
        __host__ const Type& operator[](const int idx) const { Assert(idx >= 0 && idx < NumElements);  return m_hostObject[idx]; }

    private:
        __host__ void CreateDeviceObject()
        {
            Assert(!cu_deviceObject);
            IsOk(cudaMalloc(&cu_deviceObject, sizeof(Type) * NumElements));
        }

        Type*      cu_deviceObject;
        Type       m_hostObject[NumElements];
    };
}