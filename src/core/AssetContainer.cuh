#pragma once

#include "AssetAllocator.cuh"
#include <map>
#include <vector>

//#define CudaImageBoundCheck

namespace Enso
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
		template<typename T>
		__global__ void KernelAssertSize(T* container, uint size)
		{
			CudaAssert(container->Size() == size);
		}
		
		template<typename ElementType>
		class AssetContainer<ElementType, typename std::enable_if<std::is_base_of<Host::Asset, ElementType>::value>::type> :
			public Host::Asset, 
			public AssetTags<Host::AssetContainer<typename ElementType::HostVariant>, Device::AssetContainer<typename ElementType::DeviceVariant>>
		{
			using SortFunctor = std::function<bool(const AssetHandle<ElementType>&, AssetHandle<ElementType>&)>;

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
			Device::AssetContainer<typename ElementType::DeviceVariant>*						cu_deviceData;
			typename Device::AssetContainer<typename ElementType::DeviceVariant>::Objects		m_deviceObjects;

			std::map<std::string, AssetHandle<ElementType>>		m_assetMap;
			SortFunctor											m_sortFunctor;

			AssetAllocator										m_allocator;

		public:
			__host__ AssetContainer(const std::string& id) : 
				Asset(id),
				m_allocator(*this), 
				cu_deviceData(nullptr)
			{
				cu_deviceData = m_allocator.InstantiateOnDevice<Device::AssetContainer<typename ElementType::DeviceVariant>>();
			}

			__host__ ~AssetContainer()
			{
				m_allocator.DestroyOnDevice(cu_deviceData);
				GuardedFreeDeviceArray(m_assetMap.size(), &m_deviceObjects.cu_data);
			}

			__host__ void SetSortFunctor(SortFunctor functor) { m_sortFunctor = functor; }

			__host__ void Clear()
			{
				m_assetMap.clear();
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

			__host__ void Push(std::vector<AssetHandle<ElementType>>& assetList)
			{
				for (auto& element : assetList)
				{
					Push(element);
				}
			}

			__host__ size_t Size() const { return m_assetMap.size(); }

			__host__ void Synchronise()
			{
				// Clean up first
				GuardedFreeDeviceArray(m_assetMap.size(), &m_deviceObjects.cu_data);

				if (!m_assetMap.empty())
				{
					// Create an array of the asset device instances ready to upload to the device
					std::vector<typename ElementType::DeviceVariant*> deviceInstArray;
					deviceInstArray.reserve(m_assetMap.size());
					
					// If this map needs to be sorted, do it now
					if (m_sortFunctor)
					{
						// FIXME: This is a hack designed to get around the limitations of ManagedArray. Make something better.
						std::vector<AssetHandle<ElementType>> hostArray;
						hostArray.reserve(m_assetMap.size());
						for (auto& asset : m_assetMap)
						{
							hostArray.push_back(asset.second);
						}

						std::sort(hostArray.begin(), hostArray.end(), m_sortFunctor);

						for (auto& asset : hostArray)
						{
							deviceInstArray.push_back(asset->GetDeviceInstance());
						}
					}
					else
					{
						for (auto& asset : m_assetMap)
						{
							deviceInstArray.push_back(asset.second->GetDeviceInstance());
						}
					}

					// Upload the array of device pointers
					GuardedAllocAndCopyToDeviceArray(&m_deviceObjects.cu_data, m_assetMap.size(), deviceInstArray.data());
					m_deviceObjects.numElements = m_assetMap.size();
				}
				else
				{
					m_deviceObjects.cu_data = nullptr;
					m_deviceObjects.numElements = 0;
				}

				// Synchronise the object list to the device
				SynchroniseObjects<Device::AssetContainer>(cu_deviceData, m_deviceObjects);
			}

			__host__ Device::AssetContainer<typename ElementType::DeviceVariant>* GetDeviceInstance()
			{
				AssertMsg(cu_deviceData, "Array has not been initialised!");
				return cu_deviceData;
			}

			__host__ Iterator begin() { return Iterator(*this, 0); }
			__host__ Iterator end() { return Iterator(*this, m_assetMap.size()); }

			__host__ virtual uint FromJson(const Json::Node& parentNode, const uint flags) override final
			{
				for (auto& asset : m_assetMap)
				{
					asset.second->FromJson(parentNode, flags);
				}

				return kRenderObjectClean;
			}

			__host__ void AssertSize(const uint size) const
			{
				KernelAssertSize << < 1, 1, 0 >> > (cu_deviceData, size);
				IsOk(cudaDeviceSynchronize());
			}
		};
	}
}