/*
    NOTES:
     - Unified is a small class that uses RAII to manage simple objects that mimic unified memory.
* */

#pragma once

#include "core/assets/AssetAllocator.cuh"

namespace Enso
{
    namespace Cuda
    {
        template<typename Type>
        class Object
        {
        private:
            Type                        m_hostData;
            Type*                       cu_deviceData;
            WeakAssetHandle<Host::Asset> parentAssetHandle;

        public:
            template<typename... Pack>
            __host__ Object(Pack... pack) 
            {
                static_assert(std::is_standard_layout<Type>::value, "Object type must have standard layout");    
                IsOk(cudaMalloc((void**)&cu_deviceData, sizeof(Type)));
                new (&m_hostData) Type(pack...);
                Upload();
            }

            __host__ ~Object()
            {
                cudaFree(cu_deviceData);
                m_hostData.~Type();
            }

            __host__ inline Type* operator->() { return &m_hostData; }
            __host__ inline const Type* operator->() const { return &m_hostData; }
            __host__ inline Type& operator*() { return m_hostData; }
            __host__ inline const Type& operator*() const { return m_hostData; }

            __host__ Type* GetDeviceData() { return cu_deviceData; }
            __host__ const Type* GetDeviceData() const { return cu_deviceData; }

            __host__ Object& operator=(const Type& hostCopy)
            {
                m_hostData = hostCopy;
                return *this;
            }

            __inline__ __host__ Type& Download()
            {
                IsOk(cudaMemcpy(&m_hostData, cu_deviceData, sizeof(Type), cudaMemcpyDeviceToHost));
                return m_hostData;
            }

            __inline__ __host__ void Upload()
            {
                IsOk(cudaMemcpy(cu_deviceData, &m_hostData, sizeof(Type), cudaMemcpyHostToDevice));
            }
        };

        // Copy to host memory and upload to device
        template<typename Type>
        __host__ inline Object<Type>& operator<<=(Object<Type>& lhs, const Type& rhs)
        {
            lhs = rhs;
            lhs.Upload();
            return lhs;
        }

        // Download from device and copy to host memory
        template<typename Type>
        __host__ inline Type& operator<<=(Type& lhs, Object<Type>& rhs)
        {
            lhs = rhs.Download();
            return lhs;
        }
    }
}