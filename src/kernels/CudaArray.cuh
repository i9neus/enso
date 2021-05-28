#pragma once

#include "CudaCommonIncludes.cuh"

//#define CudaImageBoundCheck

namespace Cuda
{
	namespace Device
	{
		template<typename T>
		class Array
		{
		protected:
			Array() : m_data(nullptr), m_numElements(0u) {	}

		private:
			T*		m_data;
			uint	m_numElements;

		public:
			__device__ inline uint Size() const { return m_numElements; }
			__device__ inline T& operator[](const int idx) { return m_data[idx]; }
			__device__ inline const T& operator[](const int idx) const { return m_data[idx]; }
		};
	}

	namespace Host
	{
		template<typename T> 
		class Array : public Device::Array<T>, public AssetBase
		{
		private:
			Device::Array<T>*  cu_deviceArray;
			
		public:
			__host__ Array(const std::vector<T>& arrayData) : 
				cu_deviceArray(nullptr)				
			{
				m_numElements = arrayData.size();
				
				if (m_numElements == 0) { return; }

				// Allocate the array data on the device and upload the elements in the vector
				checkCudaErrors(cudaMalloc((void**)&m_data, sizeof(T) * m_numElements));
				checkCudaErrors(cudaMemcpy(m_data, arrayData.data(), sizeof(T) * m_numElements, cudaMemcpyHostToDevice));
				
				// Allocate the array class on the device and upload the host copy
				SafeCreateDeviceInstance(&cu_deviceArray, static_cast<Device::Array<T>*>(this));
			}

			__host__ virtual void OnDestroyAsset() override final
			{
				SafeFreeDeviceMemory(&m_data);
				SafeFreeDeviceMemory(&cu_deviceArray);				
			}

			__host__ Device::Array<T>* GetDeviceInstance() 
			{
				AssertMsg(cu_deviceArray, "Array has not been initialised!");
				return cu_deviceArray;  
			}
		};
	}
}