﻿/*

	NOTES:
	 - Vector<> type has two modes: host+device and device only.
	   Host+device mode allocates memory on the host and allows it be accessed like a regular vector. To mirror it on the device, Synchronise must be called.
	   Device only mode only allocates memory on the device. This is done immediately whenever resize(), grow() or clear() are called.
* */

#pragma once

#include "core/math/Math.cuh"
#include "core/assets/AssetAllocator.cuh"
#include "core/utils/Tuple.cuh"
#include <vector>

namespace Enso
{
	enum CudaVectorFlags : uint
	{
		kVectorSyncUpload = 1,
		kVectorSyncDownload = 2,

		kVectorDeviceOnly = 4,
		kVectorNoShrinkDeviceCapacity = 8
	};

	namespace Host
	{
		template<typename HostType, typename DeviceType> class VectorBase;
	}

	template<typename Type>
	struct VectorData
	{
		__device__ void Validate() const
		{
			CudaAssert(size <= capacity);
			CudaAssert((mem == nullptr && capacity == 0) || (mem != nullptr && capacity > 0));
		}

		Type* mem = nullptr;
		size_t	size = 0;
		size_t	capacity = 0;
	};

	namespace Device
	{
		template<typename Type>
		class Vector : public Device::Asset
		{
			template<typename HostType, typename DeviceType> friend class VectorBase;

		private:
			VectorData<Type> m_data;

		public:
			template<typename ItType>
			class Iterator
			{
				friend class Vector;
				Type* m_mem;
				size_t m_idx;

			private:
				Iterator(Type* mem, const int idx) : m_mem(mem), m_idx(idx) {}

			public:
				__device__ __forceinline__ Iterator& operator++() { ++m_idx; return *this; }
				__device__ __forceinline__ Iterator& operator--() { --m_m_idx; return *this; }
				__device__ __forceinline__ bool operator!=(const Iterator& other) const { return m_idx != other.m_idx; }
				__device__ __forceinline__ ItType& operator*() { return m_mem[m_idx]; }
				__device__ __forceinline__ ItType* operator->() { return &m_mem[idx]; }			
			};

		public:
			__device__ Vector() {}
			__device__ void Synchronise(const VectorData<Type>& data) { m_data = data; }

#define GuardSize CudaAssertDebugFmt(idx < m_data.size, "Device::Vector: index %u out of bounds [0, %i).", idx, m_data.size)
#define GuardNotEmpty CudaAssertDebugMsg(m_data.size != 0, "Device::Vector: can't access empty array.")

			__device__ __forceinline__ size_t size() const { return m_data.size; }
			__device__ __forceinline__ size_t capacity() const { return m_data.capacity; }
			__device__ __forceinline__ Type& operator[](const size_t idx) { GuardSize; return m_data.mem[idx]; }
			__device__ __forceinline__ const Type& operator[](const size_t idx) const { GuardSize; return m_data.mem[idx]; }

			__device__ __forceinline__ Iterator<Type> begin() { return Iterator<Type>(m_data.mem, 0); }
			__device__ __forceinline__ Iterator<Type> begin() const { return Iterator<const Type>(m_data.mem, 0, m_data.size); }
			__device__ __forceinline__ Iterator<Type> end() { return Iterator<Type>(m_data.mem, m_size); }
			__device__ __forceinline__ Iterator<Type> end() const { return Iterator<const Type>(m_data.mem, m_data.size, m_data.size); }

			__device__ __forceinline__ Type* data() { return m_data.mem; }
			__device__ __forceinline__ const Type* data() const { return m_data.mem; }
			__device__ __forceinline__ Type& back() { GuardNotEmpty; return m_data.mem[m_data.size - 1]; }
			__device__ __forceinline__ const Type& back() const { GuardNotEmpty; return m_data.mem[m_data.size - 1]; }
			__device__ __forceinline__ Type& front() { GuardNotEmpty; return m_data.mem[0]; }
			__device__ __forceinline__ const Type& front() const { GuardNotEmpty; return m_data.mem[0]; }

#undef GuardNotEmpty
#undef GuardSize
		};
	}

	namespace Host
	{
		template<typename HostType, typename DeviceType>
		class VectorBase : public Host::Asset
		{
		public:
			template<typename ItType>
			class Iterator
			{
				friend class VectorBase;
				typename std::vector<HostType>::iterator m_it;

			private:
				Iterator(typename std::vector<HostType>::iterator& it) : m_it(it) {}

			public:
				__host__ __forceinline__ Iterator& operator++() { ++m_it; return *this; }
				__host__ __forceinline__ Iterator& operator--() { --m_it; return *this; }
				__host__ __forceinline__ bool operator!=(const Iterator& other) const { return m_it != other.m_it; }
				__host__ __forceinline__ ItType& operator*() { return *m_it; }
				__host__ __forceinline__ ItType* operator->() { return &*m_it; }
			};

		protected:
			std::vector<HostType>			m_hostVector;
			Device::Vector<DeviceType>*		cu_deviceInstance;
			VectorData<DeviceType>	     	m_deviceData;
			uint							m_flags;

		protected:
			__host__ VectorBase(const Asset::InitCtx& initCtx, const size_t initSize, const uint flags) :
				Host::Asset(initCtx),
				m_flags(flags),
				cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::Vector<DeviceType>>(*this))
			{			
				static_assert(std::is_standard_layout<DeviceType>::value, "Vector device type is non-trivial.");
				
				// Pre-allocate memory if required
				if (initSize > 0)
				{ 
					ResizeImpl(initSize);
				}
			}

			__host__ virtual ~VectorBase()
			{
				DeviceResizeImpl(0, 0);
				AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
			}

			__host__ void DeviceResizeImpl(const size_t newSize, const size_t newCapacity)
			{
				Assert(newSize <= newCapacity);

				// No change? 
				if (newSize == m_deviceData.size && newCapacity == m_deviceData.capacity) { return; }

				if (newCapacity != m_deviceData.capacity)
				{
					// Deallocate the old device memory
					AssetAllocator::GuardedFreeDevice1DArray(*this, m_deviceData.capacity, &m_deviceData.mem);
					if (newCapacity != 0)
					{
						AssetAllocator::GuardedAllocDevice1DArray(*this, newCapacity, &m_deviceData.mem, 0u);
					}
				}

				m_deviceData.capacity = newCapacity;
				m_deviceData.size = newSize;
				SynchroniseObjects<Device::Vector<DeviceType>>(cu_deviceInstance, m_deviceData);
				//LegacySynchroniseObjects(cu_deviceInstance, m_deviceData);
			}

			__host__ void ResizeImpl(const size_t newSize)
			{
				if (m_flags & kVectorDeviceOnly)
				{
					// Memory in shrinkless vectors isn't reallocated when the data get smaller
					const size_t newCapacity = (m_flags & kVectorNoShrinkDeviceCapacity && newSize < m_deviceData.capacity) ? m_deviceData.capacity : newSize;
					
					 // If memory has only been allocated on the device, immediately allocate and sync it here.
					DeviceResizeImpl(newSize, newCapacity);
				}
				else
				{
					// If this is a host+device vector, resize the host and defer synchronisation until later.
					m_hostVector.resize(newSize);
				}
			}

		public:
			// Capacity
			__host__ __forceinline__ size_t size() const { return (m_flags & kVectorDeviceOnly) ? m_deviceData.size : m_hostVector.size(); }
			__host__ __forceinline__ size_t capacity() const { return (m_flags & kVectorDeviceOnly) ? m_deviceData.capacity : m_hostVector.capacity(); }
			__host__ __forceinline__ bool empty() const { return size() == 0; }

			// Element access
			__host__ __forceinline__ HostType& operator[](const size_t idx) { Assert(!(m_flags & kVectorDeviceOnly)); return m_hostVector[idx]; }
			__host__ __forceinline__ const HostType& operator[](const size_t idx) const { Assert(!(m_flags & kVectorDeviceOnly)); return m_hostVector[idx]; }
			__host__ __forceinline__ HostType& back() { Assert(!(m_flags & kVectorDeviceOnly)); Assert(!m_hostVector.empty()); return m_hostVector.back(); }
			__host__ __forceinline__ const HostType& back() const { Assert(!(m_flags & kVectorDeviceOnly)); Assert(!m_hostVector.empty()); return m_hostVector.back(); }
			__host__ __forceinline__ HostType& front() { Assert(!(m_flags & kVectorDeviceOnly)); Assert(!m_hostVector.empty()); return m_hostVector.front(); }
			__host__ __forceinline__ const HostType& front() const { Assert(!(m_flags & kVectorDeviceOnly)); Assert(!m_hostVector.empty()); return m_hostVector.front(); }

			// Modifiers
			__host__ __forceinline__ void clear() { ResizeImpl(0u); }
			__host__ void resize(const size_t newSize) { ResizeImpl(newSize); }

			// Grow the array so that it's at least as big as newSize
			__host__ size_t grow(const size_t newSize)
			{
				if (newSize > size())
				{
					const size_t delta = newSize - size();
					ResizeImpl(newSize);
					return delta;
				}
				return 0;
			}

			__host__  void reserve(const size_t capacity)
			{
				AssertMsgFmt(!(m_flags & kVectorDeviceOnly), "Can't reserve host vector '%s' because it was initialised with kVectorDeviceOnly.", GetAssetID());
				
				return m_hostVector.reserve(capacity);
			}

			//template<typename... Pack> __host__ __forceinline__ void emplace_back(Pack... pack) { m_hostVector.emplace_back(pack...); }
			__host__ __forceinline__ void push_back(const HostType& newElement) 
			{ 
				AssertMsgFmt(!(m_flags & kVectorDeviceOnly), "Can't push_back() to host vector '%s' because it was initialised with kVectorDeviceOnly.", GetAssetID());
				
				m_hostVector.push_back(newElement); 
			}
			__host__ __forceinline__ void insert(const std::vector<HostType>& container) 
			{ 
				AssertMsgFmt(!(m_flags & kVectorDeviceOnly), "Can't insert() to host vector '%s' because it was initialised with kVectorDeviceOnly.", GetAssetID());
				
				m_hostVector.reserve(m_hostVector.size() + container.size());
				m_hostVector.insert(m_hostVector.end(), container.begin(), container.end());
			}

			__host__ __forceinline__ HostType pop_back()
			{
				AssertMsgFmt(!(m_flags & kVectorDeviceOnly), "Can't pop_back() from host vector '%s' because it was initialised with kVectorDeviceOnly.", GetAssetID());
				Assert(!m_hostVector.empty());
				HostType back = m_hostVector.back();
				m_hostVector.pop_back();

				return back;
			}

			// Iterators
			__host__ __forceinline__ Iterator<HostType> begin() { Assert(!(m_flags & kVectorDeviceOnly)); return Iterator<HostType>(m_hostVector.begin()); }
			__host__ __forceinline__ Iterator<const HostType> begin() const { Assert(!(m_flags & kVectorDeviceOnly)); return Iterator<const HostType>(m_hostVector.begin()); }
			__host__ __forceinline__ Iterator<HostType> end() { Assert(!(m_flags & kVectorDeviceOnly)); return Iterator<HostType>(m_hostVector.end()); }
			__host__ __forceinline__ Iterator<const HostType> end() const { Assert(!(m_flags & kVectorDeviceOnly)); return Iterator<const HostType>(m_hostVector.end()); }

			// Accessors
			__host__ __forceinline__ HostType* data() { return m_hostVector.data(); }
			__host__ __forceinline__ const HostType* data() const { return m_hostVector.data(); }

			__host__ void Wipe()
			{
				IsOk(cudaMemset(m_deviceData.mem, 0, sizeof(DeviceType) * m_deviceData.size));
				std::memset(m_hostVector.data(), 0, sizeof(HostType) * m_hostVector.size());
			}

			__host__ __forceinline__ Device::Vector<DeviceType>* GetDeviceInstance() { return cu_deviceInstance; }
			__host__ __forceinline__ DeviceType* GetDeviceData() { return m_deviceData.mem; }
			__host__ __forceinline__ const DeviceType* GetDeviceData() const { return m_deviceData.mem; }
		};

		template<typename CommonType>
		class Vector : public VectorBase<CommonType, CommonType>
		{
		public:
			__host__ Vector(const Asset::InitCtx& initCtx, const size_t size = 0, const uint flags = 0) :
				VectorBase<CommonType, CommonType>(initCtx, size, flags)
			{
			}

			__host__ Vector(const Vector&) = delete;
			__host__ Vector(Vector&&) = delete;
			__host__ virtual ~Vector() = default;

			__host__ void Upload()
			{
				if (!(m_flags & kVectorDeviceOnly))
				{
					// Memory in shrinkless vectors isn't reallocated when the data get smaller
					const size_t newCapacity = (m_flags & kVectorNoShrinkDeviceCapacity && m_hostVector.size() < m_deviceData.capacity) ? m_deviceData.capacity : m_hostVector.size();

					// Make sure the device data matches the host size
					DeviceResizeImpl(m_hostVector.size(), newCapacity);

					if (!m_hostVector.empty())
					{
						// Upload the data and synchronise with the device object
						IsOk(cudaMemcpy(m_deviceData.mem, m_hostVector.data(), sizeof(CommonType) * m_hostVector.size(), cudaMemcpyHostToDevice));
					}
				}
			}

			__host__ void Download()
			{
				if (m_deviceData.size > 0)
				{
					// We're assuming that the host content is already initialised
					Assert(m_deviceData.mem);
					Assert(m_hostVector.size() == m_deviceData.size);
					IsOk(cudaMemcpy(m_hostVector.data(), m_deviceData.mem, sizeof(CommonType) * m_deviceData.size, cudaMemcpyDeviceToHost));
				}
			}

			// Resizes the vector to bufferSize and copies data into it
			__host__ void SetData(const CommonType* data, const size_t bufferSize)
			{
				AssertMsg(data, "Invalid buffer pointer.");
				ResizeImpl(bufferSize);

				// Upload the data and synchronise with the device object
				IsOk(cudaMemcpy(m_deviceData.mem, data, sizeof(CommonType) * m_deviceData.size, cudaMemcpyHostToDevice));

				// Copy the data into the 
				if (!(m_flags & kVectorDeviceOnly))
				{
					std::memcpy(m_hostVector.data(), data, sizeof(CommonType) * m_hostVector.size());
				}
			}
		};

		// Stores an array of asset handles of object HostType and corresponding device pointers to DeviceType
		template<typename HostType, typename DeviceType>
		class AssetVector : public VectorBase<AssetHandle<typename HostType>, DeviceType*>
		{
		public:
			__host__ AssetVector(const Asset::InitCtx& initCtx, const size_t size = 0, const uint flags = 0) : 
				VectorBase<AssetHandle<typename HostType>, DeviceType*>(initCtx, size, flags) 
			{
			}
			__host__ virtual ~AssetVector() = default;

			__host__ void Upload()
			{
				if (!(m_flags & kVectorDeviceOnly))
				{
					// Memory in shrinkless vectors isn't reallocated when the data get smaller
					const size_t newCapacity = (m_flags & kVectorNoShrinkDeviceCapacity && m_hostVector.size() < m_deviceData.capacity) ? m_deviceData.capacity : m_hostVector.size();

					// Make sure the device data matches the host size
					DeviceResizeImpl(m_hostVector.size(), newCapacity);

					if (!m_hostVector.empty())
					{
						std::vector<DeviceType*> devicePtrs(m_hostVector.size());
						for (uint idx = 0; idx < m_hostVector.size(); ++idx)
						{
							AssertMsgFmt(m_hostVector[idx], "AssetVector '%s' has invalid asset handle at index %i", GetAssetID(), idx);
							devicePtrs[idx] = m_hostVector[idx]->GetDeviceInstance();
						}

						IsOk(cudaMemcpy(m_deviceData.mem, devicePtrs.data(), sizeof(DeviceType*) * devicePtrs.size(), cudaMemcpyHostToDevice));
					}
				}
			}

			// Helper function to downcast compatible handles to their base class pointers
			template <typename OtherType, typename = typename std::enable_if_t<std::is_base_of<HostType, OtherType>::value>>
			__host__ __forceinline__ void push_back(const AssetHandle<OtherType>& newElement) { m_hostVector.push_back(AssetHandle<HostType>(newElement)); }
		};
	}

} // namespace Enso