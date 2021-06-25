#pragma once

#include "CudaCommonIncludes.cuh"
#include <map>
#include <vector>

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
			ElementType**		    cu_data;
			uint					m_numElements;

		public:
			__host__ AssetContainer() : cu_data(nullptr), m_numElements(0) {}
			__device__ AssetContainer(ElementType** data, uint numElements) : cu_data(data), m_numElements(numElements) {	}

			__device__ __forceinline__ uint Size() const { return m_numElements; }
			__device__ __forceinline__ ElementType* operator[](const int idx) { return cu_data[idx]; }
			__device__ __forceinline__ const ElementType* operator[](const int idx) const { return cu_data[idx]; }
		};
	}

	namespace Host
	{		
		template<typename ElementType>
		class AssetContainer<ElementType, typename std::enable_if<std::is_base_of<Host::Asset, ElementType>::value>::type> :
			public Host::Asset, 
			public AssetTags<Host::AssetContainer<typename ElementType::HostVariant>, Device::AssetContainer<typename ElementType::DeviceVariant>>
		{
		public:
			class Iterator
			{
			public:
				__host__ Iterator(std::map<std::string, AssetHandle<ElementType>>::iterator& it) : m_it(it) {}

				__host__ inline bool operator != (const Iterator& other) const { return m_it != other.m_it; }
				__host__ inline AssetHandle<ElementType>& operator* () { return *m_it; }
				__host__ inline Iterator& operator++() { ++m_it; return *this; }

			private:
				std::map<std::string, AssetHandle<ElementType>>::iterator m_it;
			};

		private:
			Device::AssetContainer<typename ElementType::DeviceVariant>*	cu_deviceData;
			Device::AssetContainer<typename ElementType::DeviceVariant>		m_hostData;

			std::map<std::string, AssetHandle<ElementType>>					m_assetMap;

		public:
			__host__ AssetContainer() : cu_deviceData(nullptr) {}

			__host__ AssetHandle<ElementType> Find(const std::string& id)
			{
				auto it = m_assetMap.find(id);
				return (it == m_assetMap.end()) ? AssetHandle<ElementType>() : *it;
			}

			__host__ void Push(AssetHandle<ElementType> newAsset)
			{
				m_assetMap.push_back(newAsset);
			}

			__host__ size_t Size() const { return m_assetMap.size(); }

			__host__ void Synchronise()
			{
				// Clean up first
				SafeFreeDeviceMemory(&m_hostData.cu_data);
				SafeFreeDeviceMemory(&cu_deviceData);

				if (m_assetMap.empty()) { return; }

				// Create an array of the asset device instances ready to upload to the device
				std::vector<typename ElementType::DeviceVariant*> hostArray;
				hostArray.reserve(m_assetMap.size()));
				for(auto& asset : m_assetMap)
				{
					hostArray.push_back(asset->GetDeviceInstance());
				}

				SafeAllocAndCopyToDeviceMemory(&m_hostData.cu_data, m_assetMap.size(), hostArray.data());

				cu_deviceData = InstantiateOnDevice<Device::AssetContainer<typename ElementType::DeviceVariant>>(m_hostData.cu_data, m_assetMap.size());
			}

			__host__ void Destroy()
			{
				for (int i = 0; i < m_assetMap.size(); i++)
				{
					m_assetMap[i].DestroyAsset();
				}

				DestroyOnDevice(&cu_deviceData);
				SafeFreeDeviceMemory(&m_hostData.cu_data);
			}

			__host__ virtual void OnDestroyAsset() override final
			{
				Destroy();
			}

			__host__ Device::AssetContainer<typename ElementType::DeviceVariant>* GetDeviceInstance()
			{
				AssertMsg(cu_deviceData, "Array has not been initialised!");
				return cu_deviceData;
			}

			__host__ Iterator begin() { return Iterator(*this, 0); }
			__host__ Iterator end() { return Iterator(*this, m_assetMap.size()); }

			__host__ virtual void OnJson(const Json::Node& parentNode) override final
			{
				for (auto& asset : m_assetMap)
				{
					asset->OnJson(parentNode);
				}
			}
		};
	}
}