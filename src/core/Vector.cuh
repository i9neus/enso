#pragma once

#include "math/Math.cuh"
#include "AssetAllocator.cuh"
#include "Tuple.cuh"
#include <vector>

namespace Enso
{
	enum CudaVectorFlags : uint
	{
		kVectorSyncUpload = 1,
		kVectorSyncDownload = 2
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
				__device__ __forceinline__ ItType* operator->() { return &m_mem[idx]; }			};

		public:
			__device__ Vector() {}
			__device__ void Synchronise(const VectorData<Type>& data) { m_data = data; }


			__device__ __forceinline__ size_t size() const { return m_data.size; }
			__device__ __forceinline__ Type& operator[](const int idx) { CudaAssertDebug(idx < m_data.size); return m_data.mem[idx]; }
			__device__ __forceinline__ const Type& operator[](const int idx) const { CudaAssertDebug(idx < m_data.size); return m_data.mem[idx]; }

			__device__ __forceinline__ Iterator<Type> begin() { return Iterator<Type>(m_data.mem, 0); }
			__device__ __forceinline__ Iterator<Type> begin() const { return Iterator<const Type>(m_data.mem, 0, m_data.size); }
			__device__ __forceinline__ Iterator<Type> end() { return Iterator<Type>(m_data.mem, m_size); }
			__device__ __forceinline__ Iterator<Type> end() const { return Iterator<const Type>(m_data.mem, m_data.size, m_data.size); }

			__device__ __forceinline__ Type* data() { return m_data.mem; }
			__device__ __forceinline__ const Type* data() const { return m_data.mem; }
			__device__ __forceinline__ Type& back() { CudaAssertDebug(m_data.size != 0); return m_data.mem[m_data.size - 1]; }
			__device__ __forceinline__ const Type& back() const { CudaAssertDebug(m_data.size != 0); return m_data.mem[m_data.size - 1]; }
			__device__ __forceinline__ Type& front() { CudaAssertDebug(m_data.size != 0); return m_data.mem[0]; }
			__device__ __forceinline__ const Type& front() const { CudaAssertDebug(m_data.size != 0); return m_data.mem[0]; }
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
			std::vector<HostType>			m_hostData;
			//Device::Vector<DeviceType>*	m_hostInstance;
			Device::Vector<DeviceType>*		cu_deviceInstance;
			VectorData<DeviceType>	     	m_deviceData;

		protected:
			__host__ VectorBase(const Asset::InitCtx& initCtx, const size_t size) :
				Host::Asset(initCtx),
				cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::Vector<DeviceType>>(*this))
			{
				if (size > 0)
				{
					DestructiveResizeImpl(size);
				}
			}

			__host__ virtual ~VectorBase()
			{
				DestructiveResizeImpl(0);
				AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
			}

			__host__ VectorBase(const size_t size) : m_hostData(size) {}

			__host__ void DestructiveResizeImpl(const size_t newSize)
			{
				// Deallocate the old device memory
				AssetAllocator::GuardedFreeDeviceArray(*this, m_deviceData.capacity, &m_deviceData.mem);

				if (newSize == 0)
				{
					m_deviceData.mem = nullptr;
					m_deviceData.capacity = 0;
					m_deviceData.size = 0;
				}
				else
				{
					AssetAllocator::GuardedAllocDeviceArray(*this, newSize, &m_deviceData.mem, 0u);
					m_deviceData.capacity = newSize;
					m_deviceData.size = newSize;
				}
			}

		public:
			// Capacity
			__host__ __forceinline__ size_t size() const { return m_hostData.size(); }
			__host__ __forceinline__ size_t capacity() const { return m_hostData.capacity(); }
			__host__ __forceinline__ void reserve(const int capacity) { return m_hostData.reserve(capacity); }
			__host__ __forceinline__ bool empty() const { return m_hostData.empty(); }

			// Element access
			__host__ __forceinline__ HostType& operator[](const int idx) { return m_hostData[idx]; }
			__host__ __forceinline__ const HostType& operator[](const int idx) const { return m_hostData[idx]; }
			__host__ __forceinline__ HostType& back() { Assert(!m_hostData.empty()); return m_hostData.back(); }
			__host__ __forceinline__ const HostType& back() const { Assert(!m_hostData.empty()); return m_hostData.back(); }
			__host__ __forceinline__ HostType& front() { Assert(!m_hostData.empty()); return m_hostData.front(); }
			__host__ __forceinline__ const HostType& front() const { Assert(!m_hostData.empty()); return m_hostData.front(); }

			// Modifiers
			__host__ __forceinline__ void clear() { m_hostData.clear(); }
			__host__ __forceinline__ void resize(const size_t newSize) { m_hostData.resize(newSize); }
			__host__ __forceinline__ void grow(const int deltaSize)
			{
				const int newSize = int(m_hostData.size()) + deltaSize;
				AssertMsgFmt(newSize >= 0, "Growing vector '%s' by %i elements results in negative size.", GetAssetID(), deltaSize);
				m_hostData.resize(newSize);
			}
			//template<typename... Pack> __host__ __forceinline__ void emplace_back(Pack... pack) { m_hostData.emplace_back(pack...); }
			__host__ __forceinline__ void push_back(const HostType& newElement) { m_hostData.push_back(newElement); }

			__host__ __forceinline__ HostType pop_back()
			{
				Assert(!m_hostData.empty());
				HostType back = m_hostData.back();
				m_hostData.pop_back();
				return back;
			}

			// Iterators
			__host__ __forceinline__ Iterator<HostType> begin() { return Iterator<HostType>(m_hostData.begin()); }
			__host__ __forceinline__ Iterator<const HostType> begin() const { return Iterator<const HostType>(m_hostData.begin()); }
			__host__ __forceinline__ Iterator<HostType> end() { return Iterator<HostType>(m_hostData.end()); }
			__host__ __forceinline__ Iterator<const HostType> end() const { return Iterator<const HostType>(m_hostData.end()); }

			// Accessors
			__host__ __forceinline__ HostType* data() { return m_hostData.data(); }
			__host__ __forceinline__ const HostType* data() const { return m_hostData.data(); }

			__host__ void Wipe()
			{
				IsOk(cudaMemset(m_deviceData.mem, 0, sizeof(DeviceType) * m_deviceData.size));
				std::memset(m_hostData.data(), 0, sizeof(HostType) * m_hostData.size());
			}

			__host__ __forceinline__ Device::Vector<DeviceType>* GetDeviceInstance() { return cu_deviceInstance; }
		};

		template<typename CommonType>
		class Vector : public VectorBase<CommonType, CommonType>
		{
		public:
			__host__ Vector(const Asset::InitCtx& initCtx, const size_t size = 0) : VectorBase<CommonType, CommonType>(initCtx, size) {}
			__host__ virtual ~Vector() = default;

			__host__ void Synchronise(const uint syncFlags)
			{
				if (syncFlags == kVectorSyncUpload)
				{
					// Make sure the device data matches the host size
					if (m_hostData.size() != m_deviceData.size)
					{
						DestructiveResizeImpl(m_hostData.size());
					}

					if (!m_hostData.empty())
					{
						// Upload the data and synchronise with the device object
						IsOk(cudaMemcpy(m_deviceData.mem, m_hostData.data(), sizeof(CommonType) * m_hostData.size(), cudaMemcpyHostToDevice));
						SynchroniseObjects<Device::Vector<CommonType>>(cu_deviceInstance, m_deviceData);
					}
				}
				else
				{
					if (m_deviceData.size > 0)
					{
						// We're assuming that the host content is already initialised
						Assert(m_deviceData.mem);
						Assert(m_hostData.size() == m_deviceData.size);
						IsOk(cudaMemcpy(m_hostData.data(), m_deviceData.mem, sizeof(CommonType) * m_deviceData.size, cudaMemcpyDeviceToHost));
					}
				}
			}
		};

		// Stores an array of asset handles of object HostType and corresponding device pointers to DeviceType
		template<typename HostType, typename DeviceType>
		class AssetVector : public VectorBase<AssetHandle<typename HostType>, DeviceType*>
		{
		public:
			__host__ AssetVector(const Asset::InitCtx& initCtx, const size_t size = 0) : VectorBase<AssetHandle<typename HostType>, DeviceType*>(initCtx, size) {}
			__host__ virtual ~AssetVector() = default;

			__host__ void Synchronise(const uint syncFlags)
			{
				AssertMsg(syncFlags == kVectorSyncUpload, "Using kVectorSyncDownload on AssetVector type is meaningless.");				

				// Make sure the device data matches the host size
				if (m_hostData.size() != m_deviceData.size)
				{
					DestructiveResizeImpl(m_hostData.size());
				}

				if (!m_hostData.empty())
				{
					std::vector<DeviceType*> devicePtrs(m_hostData.size());
					for (int idx = 0; idx < m_hostData.size(); ++idx)
					{
						devicePtrs[idx] = m_hostData[idx]->GetDeviceInstance();
					}

					IsOk(cudaMemcpy(m_deviceData.mem, devicePtrs.data(), sizeof(DeviceType*) * devicePtrs.size(), cudaMemcpyHostToDevice));
					LegacySynchroniseObjects(cu_deviceInstance, m_deviceData);
				}
			}

			// Helper function to downcast compatible handles to their base class pointers
			template <typename OtherType, typename = typename std::enable_if_t<std::is_base_of<HostType, OtherType>::value>>
			__host__ __forceinline__ void push_back(const AssetHandle<OtherType>& newElement) { m_hostData.push_back(AssetHandle<HostType>(newElement)); }
		};
	}

} // namespace Enso