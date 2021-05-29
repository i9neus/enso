#pragma once

#include "CudaCommonIncludes.cuh"

//#define CudaImageBoundCheck

namespace Cuda
{
	namespace Device
	{
		template<typename T>
		class AssetContainer : public ManagedPair<Device::AssetContainer<T>>
		{
		protected:
			AssetContainer() : cu_data(nullptr), m_numElements(0u) {	}

			T*		cu_data;
			uint	m_numElements;

		public:
			__device__ inline uint Size() const { return m_numElements; }
			__device__ inline T& operator[](const int idx) { return cu_data[idx]; }
			__device__ inline const T& operator[](const int idx) const { return cu_data[idx]; }
		};
	}

	namespace Host
	{
		template<typename ElementType, typename = std::enable_if<std::is_base_of<ManagedPair<ElementType>, ElementType>::value>>
		class AssetContainer : public Device::AssetContainer<typename ElementType::DeviceVariant>, public AssetBase
		{
		private:
			Device::AssetContainer<typename ElementType::DeviceVariant>* cu_deviceContainer;

			std::vector<Asset<ElementType>> m_assets;

		public:
			__host__ AssetContainer() : cu_deviceContainer(nullptr) {}

			__host__ void Push(Asset<ElementType> newAsset)
			{
				m_assets.push_back(newAsset);
			}

			__host__ size_t Size() const { return m_assets.size(); }

			__host__ void Sync()
			{
				// Clean up first
				SafeFreeDeviceMemory(&cu_data);
				SafeFreeDeviceMemory(&cu_deviceContainer);

				m_numElements = m_assets.size();
				if (m_numElements == 0) { return; }

				// Create an array of the asset device instances ready to upload to the device
				std::vector<typename ElementType::DeviceVariant*> hostArray(m_numElements);
				const size_t hostArraySize = sizeof(typename ElementType::DeviceVariant*) * m_numElements;
				for (int i = 0; i < m_numElements; i++)
				{
					hostArray[i] = m_assets[i]->GetDeviceInstance();
				}

				// Allocate the array data on the device and upload the elements in the vector
				checkCudaErrors(cudaMalloc((void**)&cu_data, hostArraySize));
				checkCudaErrors(cudaMemcpy(cu_data, hostArray.data(), hostArraySize, cudaMemcpyHostToDevice));

				// Allocate the array class on the device and upload the host copy
				SafeCreateDeviceInstance(&cu_deviceContainer, static_cast<Device::AssetContainer<typename ElementType::DeviceVariant>*>(this));
			}

			__host__ void Clear()
			{
				for (int i = 0; i < m_assets.size(); i++)
				{
					m_assets[i].DestroyAsset();
				}

				SafeFreeDeviceMemory(&cu_data);
				SafeFreeDeviceMemory(&cu_deviceContainer);
			}

			__host__ virtual void OnDestroyAsset() override final
			{
				Clear();
			}

			__host__ Device::AssetContainer<typename ElementType::DeviceVariant>* GetDeviceInstance()
			{
				AssertMsg(cu_deviceContainer, "Array has not been initialised!");
				return cu_deviceContainer;  
			}
		};
	}
}