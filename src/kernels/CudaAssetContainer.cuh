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
		public:
			struct Objects
			{
				__device__ Objects() : cu_data(nullptr), numElements(0) {}
				__device__ Objects(ElementType** data_, uint numElements_) : cu_data(data_), numElements(numElements_) {}
				ElementType**	cu_data;
				uint			numElements;
			};

		protected:
			Objects		m_objects;

		public:
			__device__ AssetContainer() = default;
			__device__ __forceinline__ uint Size() const { return m_objects.numElements; }
			__device__ __forceinline__ ElementType* operator[](const int idx) { return m_objects.cu_data[idx]; }
			__device__ __forceinline__ const ElementType* operator[](const int idx) const { return m_objects.cu_data[idx]; }
			
			__device__ void Synchronise(const Objects& objects) 
			{ 
				m_objects = objects; 
			}
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
				__host__ Iterator(typename std::map<std::string, AssetHandle<ElementType>>::iterator& it) : m_it(it) {}

				__host__ inline bool operator != (const Iterator& other) const { return m_it != other.m_it; }
				__host__ inline AssetHandle<ElementType>& operator* () { return *m_it; }
				__host__ inline Iterator& operator++() { ++m_it; return *this; }

			private:
				typename std::map<std::string, AssetHandle<ElementType>>::iterator m_it;
			};

		private:
			Device::AssetContainer<typename ElementType::DeviceVariant>*				cu_deviceData;
			typename Device::AssetContainer<typename ElementType::DeviceVariant>::Objects		m_deviceObjects;

			std::map<std::string, AssetHandle<ElementType>>					m_assetMap;

		public:
			__host__ AssetContainer() : cu_deviceData(nullptr) 
			{
				cu_deviceData = InstantiateOnDevice<Device::AssetContainer<typename ElementType::DeviceVariant>>();
			}

			__host__ AssetHandle<ElementType> Find(const std::string& id)
			{
				auto it = m_assetMap.find(id);
				return (it == m_assetMap.end()) ? AssetHandle<ElementType>() : *it;
			}

			__host__ void Push(AssetHandle<ElementType>& newAsset)
			{
				m_assetMap[newAsset->GetAssetID()] = newAsset;
			}

			__host__ size_t Size() const { return m_assetMap.size(); }

			__host__ virtual void Synchronise() override final
			{
				// Clean up first
				SafeFreeDeviceMemory(&m_deviceObjects.cu_data);

				if (!m_assetMap.empty())
				{
					// Create an array of the asset device instances ready to upload to the device
					std::vector<typename ElementType::DeviceVariant*> hostArray;
					hostArray.reserve(m_assetMap.size());
					for (auto& asset : m_assetMap)
					{
						hostArray.push_back(asset.second->GetDeviceInstance());
					}

					// Upload the array of device pointers
					SafeAllocAndCopyToDeviceMemory(&m_deviceObjects.cu_data, m_assetMap.size(), hostArray.data());
					m_deviceObjects.numElements = m_assetMap.size();
				}
				else
				{
					m_deviceObjects.cu_data = nullptr;
					m_deviceObjects.numElements = 0;
				}

				// Synchronise the object list to the device
				SynchroniseObjects(cu_deviceData, m_deviceObjects);
			}

			__host__ void Destroy()
			{
				DestroyOnDevice(&cu_deviceData);
				SafeFreeDeviceMemory(&m_deviceObjects.cu_data);
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

			__host__ virtual void FromJson(const ::Json::Node& parentNode, const uint flags) override final
			{
				for (auto& asset : m_assetMap)
				{
					asset.second->FromJson(parentNode, flags);
				}
			}
		};
	}
}