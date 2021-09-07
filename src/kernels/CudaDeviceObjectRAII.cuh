#pragma once

#include "CudaCommonIncludes.cuh"

namespace Cuda
{
    // Tiny wrapper that wraps an object and keeps a 
    template<typename T, typename = typename std::enable_if<std::is_standard_layout<T>::value>>
    class DeviceObjectRAII
    {
    public:
        __host__ DeviceObjectRAII() : cu_deviceObject(nullptr) {}

        __host__ ~DeviceObjectRAII()
        {
            IsOk(cudaFree(cu_deviceObject));
        }

        __host__ DeviceObjectRAII(const T& object) : 
            DeviceObjectRAII()
        {
            Upload(object);
        }

        __host__ DeviceObjectRAII& operator=(const T& object)
        {   
            Upload(object);
        }

        __host__ void Upload(const T& object)
        {
            m_hostObject = object;
            Upload();
        }

        __host__ void Upload()
        {
            if (!cu_deviceObject)
            {
                CreateDeviceObject();
            }
            IsOk(cudaMemcpy(cu_deviceObject, &m_hostObject, sizeof(T), cudaMemcpyHostToDevice));
        }

        __host__ T& Download()
        {
            if (cu_deviceObject)
            {
                IsOk(cudaMemcpy(&m_hostObject, cu_deviceObject, sizeof(T), cudaMemcpyDeviceToHost));
            }
            return m_hostObject;
        }

        __host__ T* GetDeviceObject()
        {
            if (!cu_deviceObject)
            {
                CreateDeviceObject();
            }
            return cu_deviceObject;
        }

        __host__ T* operator->(void) { return &m_hostObject; }
        __host__ T& GetHostObject() { return m_hostObject; }

    private:
        __host__ void CreateDeviceObject()
        {
            Assert(!cu_deviceObject);
            IsOk(cudaMalloc(&cu_deviceObject, sizeof(T)));
        }

        T*      cu_deviceObject;
        T       m_hostObject;
    };
}