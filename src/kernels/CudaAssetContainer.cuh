#pragma once

#include "CudaCommonIncludes.cuh"

//#define CudaImageBoundCheck

namespace Cuda
{
	namespace Host
	{
		template<typename ElementType, typename Enable = void> class AssetContainer;
	}
	
	namespace Device
	{
		template<typename ElementType>
		class AssetContainer : public Device::Asset, public AssetTags<Host::AssetContainer<ElementType>, Device::AssetContainer<ElementType>>
		{
			template<typename T, typename S> friend class Host::AssetContainer;

		protected:
			AssetContainer() : cu_data(nullptr), m_numElements(0u) {	}

			ElementType**	cu_data;
			uint			m_numElements;

		public:
			__device__ inline uint Size() const { return m_numElements; }
			__device__ inline ElementType& operator[](const int idx) { return *cu_data[idx]; }
			__device__ inline const ElementType& operator[](const int idx) const { return *cu_data[idx]; }
		};
	}

	namespace Host
	{		
		template<typename ElementType>
		class AssetContainer<ElementType, typename std::enable_if<std::is_base_of<Host::Asset, ElementType>::value>::type> :
			public Host::Asset, 
			public AssetTags<Host::AssetContainer<typename ElementType::HostVariant>, Device::AssetContainer<typename ElementType::DeviceVariant>>
		{
		private:
			Device::AssetContainer<typename ElementType::DeviceVariant>* cu_deviceData;
			Device::AssetContainer<typename ElementType::DeviceVariant> m_hostData;

			std::vector<AssetHandle<ElementType>> m_assets;

		public:
			__host__ AssetContainer() : cu_deviceData(nullptr) {}

			__host__ void Push(AssetHandle<ElementType> newAsset)
			{
				m_assets.push_back(newAsset);
			}

			__host__ size_t Size() const { return m_assets.size(); }

			__host__ void Sync()
			{
				// Clean up first
				SafeFreeDeviceMemory(&m_hostData.cu_data);
				SafeFreeDeviceMemory(&cu_deviceData);

				m_numElements = m_assets.size();
				if (m_numElements == 0) { return; }

				// Create an array of the asset device instances ready to upload to the device
				std::vector<typename ElementType::DeviceVariant*> hostArray(m_numElements);
				for (int i = 0; i < m_numElements; i++)
				{
					hostArray[i] = m_assets[i]->GetDeviceInstance();
				}

				// Allocate the array data on the device and upload the elements in the vector
				SafeAllocDeviceArray(&cu_data, m_numElements, hostArray.data());

				// Allocate the array class on the device and upload the host copy
				SafeCreateDeviceInstance(&cu_deviceContainer, static_cast<Device::AssetContainer<typename ElementType::DeviceVariant>*>(this));
			}

			__host__ void Clear()
			{
				for (int i = 0; i < m_assets.size(); i++)
				{
					m_assets[i].DestroyAsset();
				}

				DestroyOnDevice(&cu_deviceData);
				SafeFreeDeviceMemory(&m_hostData.cu_data);
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