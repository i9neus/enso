#pragma once

#include "math/CudaMath.cuh"
#include "CudaCommonIncludes.cuh"

namespace Cuda
{
	//enum class AccessSignal : uint { kUnlocked, kReadLocked, kWriteLocked };

	enum CudaVectorFlags : uint
	{
		kVectorUnifiedMemory = 1,
		kVectorHostAlloc = 2,
		kVectorSyncDeviceAlloc = 4,
		kVectorSyncUpload = 8,
		kVectorSyncDownload = 16
	};
	
	namespace Host
	{
		// Forward declare some things we'll need for friendship and tagging
		template<typename ElementType, typename HostType, typename DeviceType> class VectorBase;
		template<typename ElementType> class Vector;
	}

	struct VectorParams
	{
		uint size = 0;
		uint capacity = 0;
		uint flags = 0;
	};

	namespace Device
	{
		template<typename ElementType, typename HostType, typename DeviceType>
		class VectorBase : public Device::Asset, public AssetTags<HostType, DeviceType>
		{	
		public:
			__device__ VectorBase() :
				cu_data(nullptr) {}
			__device__ ~VectorBase() {}

			__device__ __forceinline__ unsigned int								Size() const { return m_params.size; }
			__device__ __forceinline__ unsigned int								Capacity() const { return m_params.capacity; }
			__device__ __forceinline__ bool										IsEmpty() const { return m_params.size == 0; }
			__device__ __forceinline__ unsigned int								MemorySize() const { return m_params.size * sizeof(T); }

			__device__ ElementType* Data()										{ return cu_data; }
			__device__ ElementType& operator[](const uint idx)					{ return cu_data[idx]; }
			__device__ const ElementType& operator[](const uint idx) const		{ return cu_data[idx]; }

			__device__ void Synchronise(ElementType* data, const VectorParams& params)
			{
				cu_data = data;
				m_params = params;
			}

		protected:
			ElementType*		cu_data;
			VectorParams		m_params;
		};

		// Wrapper class that hides the messy template parameterisation
		template<typename ElementType>
		class Vector : public Device::VectorBase<ElementType, Host::Vector<ElementType>, Device::Vector<ElementType>>
		{
		public:
			__host__ __device__ Vector() : VectorBase<ElementType, Host::Vector<ElementType>, Device::Vector<ElementType>>() {}
		};
	}

	namespace Host
	{
		template<typename ElementType, typename HostType, typename DeviceType>
		class VectorBase : public Host::Asset, public AssetTags<HostType, DeviceType>
		{
		protected:
			DeviceType*			cu_deviceInstance;

			ElementType*		m_hostData;
			ElementType*		cu_deviceData;

			VectorParams		m_hostParams;
			VectorParams		m_deviceParams;

			cudaStream_t		m_hostStream;
		public:
			
			template<typename Type>
			struct IteratorBase
			{
				IteratorBase(const uint idx, ElementType* data) : m_idx(idx), m_data(data) {}

				IteratorBase& operator++() { ++m_idx; return *this; }
				bool operator!=(const IteratorBase& other) const { return m_idx != other.m_idx; }

				ElementType& operator*() { return m_data[m_idx]; }
				ElementType& operator*() const { return m_data[m_idx]; }
				ElementType* operator->() { return &m_data[m_idx]; }
				ElementType* operator->() const { return &m_data[m_idx]; }

			private:	
				ElementType*	m_data;
				uint			m_idx;
			};

			using Iterator = IteratorBase<ElementType>;
			using ConstIterator = IteratorBase<const ElementType>;

			/*struct Iterator : IteratorBase<Iterator>
			{
				ElementType& operator*() { return m_data[m_idx]; }
				Iterator(const uint idx, ElementType* data) : IteratorBase<Iterator>(idx, data) {}
			};
			struct ConstIterator : IteratorBase<ConstIterator>
			{
				const ElementType& operator*() const { return m_data[m_idx]; }
				ConstIterator(const uint idx, ElementType* data) : IteratorBase<ConstIterator>(idx, data) {}
			};*/

		public:
			__host__ VectorBase(const std::string& id, const uint flags, cudaStream_t hostStream) :
				Asset(id),
				cu_deviceInstance(nullptr),
				cu_deviceData(nullptr),
				m_hostData(nullptr)
			{
				m_hostStream = hostStream;
				m_hostParams.flags = flags;

				AssertMsg(!((m_hostParams.flags & kVectorUnifiedMemory) && !(m_hostParams.flags & kVectorHostAlloc)),
					"Must specify kVectorHostAlloc when using kVectorUnifiedMemory.");

				// Create a device instance
				cu_deviceInstance = InstantiateOnDevice<DeviceType>(id);
			}

			__host__ VectorBase(const std::string& id, const uint size, const uint flags, cudaStream_t hostStream) :
				Host::VectorBase<ElementType, HostType, DeviceType>(id, flags, hostStream)
			{
				// Allocate and sync the memory
				ResizeImpl(size, true, false);
			}

			__host__  virtual ~VectorBase()
			{
				OnDestroyAsset();
			}

			__host__  virtual void OnDestroyAsset() override final
			{
				DestroyOnDevice(GetAssetID(), cu_deviceInstance);
				GuardedFreeDeviceArray(GetAssetID(), m_deviceParams.capacity, &cu_deviceData);

				if (m_hostData && !(m_hostParams.flags & kVectorUnifiedMemory))
				{
					delete[] m_hostData;
					m_hostData = nullptr;
				}
			}

			__host__ inline void Prepare()
			{
				// FIXME: On Windows, CUDA requires that cudaDeviceSynchronize() is called after kernels access shared memory. 
				if (m_hostParams.flags & kVectorUnifiedMemory)
				{
					IsOk(cudaDeviceSynchronize());
				}
			}

			__host__ inline Iterator		begin() { return Iterator(0, m_hostData);  }
			__host__ inline Iterator		end() { return Iterator(m_hostParams.size, m_hostData); }
			__host__ inline ConstIterator	begin() const { return ConstIterator(0, m_hostData); }
			__host__ inline ConstIterator	end() const { return ConstIterator(m_hostParams.size, m_hostData); }

			__host__ inline DeviceType*		GetDeviceInstance() const { return cu_deviceInstance; }
			__host__ inline ElementType*	GetHostData()
			{
				AssertMsgFmt(m_hostParams.flags & kVectorHostAlloc, "Vector '%s' does not have host allocation.", GetAssetID().c_str());
				return m_hostData;
			}

			__host__ inline uint			Size() const { return m_hostParams.size; }
			__host__ inline uint			Capacity() const { return m_hostParams.capacity; }
			__host__ inline bool			IsEmpty() const { return m_hostParams.size == 0; }

			__host__ inline const ElementType& operator[](const uint idx) const
			{
				return const_cast<VectorBase*>(this)->operator[](idx);
			}

			__host__ ElementType& operator[](const uint idx)
			{
				AssertMsgFmt(m_hostParams.flags & kVectorHostAlloc, "Vector '%s': trying to use [] on array that does not have host allocation.", GetAssetID().c_str());
				Assert(idx < m_hostParams.size);

				return m_hostData[idx];
			}

			// ---------------------------------------------------------------------------------------------------------------------------------------------	
			
			__host__ inline void Reserve(const uint newCapacity)	{ ReserveImpl(newCapacity, m_hostParams.flags & kVectorSyncDeviceAlloc, true);	}
			__host__ inline void Resize(const uint newSize)			{ ResizeImpl(newSize, m_hostParams.flags & kVectorSyncDeviceAlloc, true); }

			__host__ inline void Grow(const uint inc)				{ ResizeImpl(m_hostParams.size + inc, m_hostParams.flags & kVectorSyncDeviceAlloc, true); }
			__host__ inline void Shrink(const uint dec)				{ ResizeImpl(m_hostParams.size - dec, m_hostParams.flags & kVectorSyncDeviceAlloc, true); }

			__host__ void PushBack(const ElementType& element)
			{
				AssertMsg(m_hostParams.flags & kVectorHostAlloc, "Calling PushBack() on a Vector that does not have host allocation.");
				Resize(m_hostParams.size + 1);
				m_hostData[m_hostParams.size - 1] = element;
			}

			template<typename... Pack>
			__host__ void EmplaceBack(Pack... pack)
			{
				AssertMsg(m_hostParams.flags & kVectorHostAlloc, "Calling EmplaceBack() on a Vector that does not have host allocation.");
				Resize(m_hostParams.size + 1);
				m_hostData[m_hostParams.size - 1] = ElementType(pack...);
			}

			__host__ void PopBack()
			{
				AssertMsg(m_hostParams.flags & kVectorHostAlloc, "Calling PopBack() on a Vector that does not have host allocation.");
				Resize(m_hostParams.size - 1);
			}

			__host__ inline ElementType& Back()
			{
				AssertMsg(m_hostParams.flags & kVectorHostAlloc, "Calling Back() on a Vector that does not have host allocation.");
				return m_hostData[m_hostParams.size - 1];
			}

			__host__ inline ElementType& Front()
			{
				AssertMsg(m_hostParams.flags & kVectorHostAlloc, "Calling Front() on a Vector that does not have host allocation.");
				return m_hostData[0];
			}

			__host__ inline void Clear()
			{
				Resize(0);
			}

			// Transfers the contents of the buffers between the host and device 
			__host__ void Synchronise(const uint syncFlags)
			{
				AssertMsg(!(m_hostParams.flags & kVectorUnifiedMemory), "Calling Upload() on a Vector with kVectorUnifiedMemory flag set.");
				AssertMsg(m_hostParams.flags & kVectorHostAlloc, "Calling Upload() on a Vector that does not have host allocation.");
				AssertMsg(syncFlags & (kVectorSyncUpload | kVectorSyncDownload), "Invalid syncronisation flags.");

				if (syncFlags == kVectorSyncUpload)
				{
					// Make sure the device data matches the host size
					if (m_hostParams.size != m_deviceParams.size)
					{
						ResizeImpl(m_hostParams.size, true, false);
					}
					
					if (cu_deviceData)
					{
						IsOk(cudaMemcpy(cu_deviceData, m_hostData, sizeof(ElementType) * m_hostParams.size, cudaMemcpyHostToDevice));
						Cuda::Synchronise(cu_deviceInstance, cu_deviceData, m_deviceParams);
					}
				}
				else
				{
					if (cu_deviceData)
					{
						IsOk(cudaMemcpy(m_hostData, cu_deviceData, sizeof(ElementType) * m_hostParams.size, cudaMemcpyDeviceToHost));
					}
				}
			}

			// Erases the contents of the vector with the specified value
			__host__ void Fill(const ElementType& value)
			{
				Assert(cu_deviceInstance);

				constexpr int blockSize = 16 * 16;
				int gridSize = (m_deviceParams.size + (m_blockSize - 1)) / (m_blockSize);
				KernelFill << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance, value);

				// If the vector has host allocation, reset that too
				if (m_hostParams.flags & kVectorHostAlloc && !(m_hostParams.flags & kVectorUnifiedMemory))
				{
					for (int idx = 0; idx < m_hostParams.size; ++idx)
					{
						m_hostData[idx] = value;
					}
				}

				IsOk(cudaDeviceSynchronize());
			}

			// Zeroes the contents of the vector
			__host__ void Wipe()
			{
				Assert(cu_deviceInstance);

				IsOk(cudaMemset(m_hostData, 0, sizeof(ElementType) * m_hostParams.size));
				IsOk(cudaStreamSynchronize(m_hostStream));

				// If the vector has host allocation, reset that too
				if (m_hostParams.flags & kVectorHostAlloc && !(m_hostParams.flags & kVectorUnifiedMemory))
				{
					std::memset(m_hostData, 0, sizeof(ElementType) * m_hostParams.size);
				}
			}

		private:
			__host__ void ReserveImpl(const uint newCapacity, const bool deviceAlloc, const bool deviceCopy)
			{			
				// If the device is being explicitly synced or we're using unified memory, realloc the device memory 
				if (newCapacity > m_deviceParams.capacity && (deviceAlloc || (m_hostParams.flags & kVectorUnifiedMemory)))
				{
					// Don't adjust capacity if it means reducing the size of the array
					Assert(newCapacity >= m_deviceParams.size);
					
					// Allocate a new block of memory
					ElementType* newDeviceData = nullptr;
					GuardedAllocDeviceArray(GetAssetID(), newCapacity, &newDeviceData, (m_hostParams.flags & kVectorUnifiedMemory) ? kCudaMemoryManaged : 0u);

					// If we're syncing from host to device, don't copy the data from the old device buffer because it'll just get overwritten anyway
					if (cu_deviceData)
					{
						if (deviceCopy || (m_hostParams.flags & kVectorUnifiedMemory))
						{
							IsOk(cudaMemcpy(newDeviceData, cu_deviceData, m_hostParams.size * sizeof(ElementType), cudaMemcpyDeviceToDevice));
						}

						// Deallocate the old memory
						GuardedFreeDeviceArray(GetAssetID(), m_hostParams.capacity, &cu_deviceData);
					}

					m_deviceParams.capacity = newCapacity;
					cu_deviceData = newDeviceData;

					Log::System("Capacity of device vector '%s' changed to %i.", GetAssetID().c_str(), m_hostParams.capacity);
				}

				// Unified memory is accessible on both host and and device so just copy the pointer
				if (m_hostParams.flags & kVectorUnifiedMemory)
				{					
					m_hostData = cu_deviceData;
					m_hostParams.capacity = newCapacity;
				}
				// Otherwise, reallocate the host data
				else if (m_hostParams.flags & kVectorHostAlloc && newCapacity > m_hostParams.capacity)
				{
					// Don't adjust capacity if it means reducing the size of the array
					Assert(newCapacity >= m_hostParams.size);
					
					ElementType* newHostData = new ElementType[newCapacity];
					if (m_hostData)
					{
						std::memcpy(newHostData, m_hostData, m_hostParams.size * sizeof(ElementType));
					}

					delete[] m_hostData;
					m_hostData = newHostData;
					m_hostParams.capacity = newCapacity;

					Log::System("Capacity of host vector '%s' changed to %i.", GetAssetID().c_str(), m_hostParams.capacity);
				}
			}

			__host__ void ResizeImpl(const uint newSize, const bool deviceAlloc, const bool deviceCopy)
			{
				// Don't resize if nothing has changed
				if (m_hostParams.size == newSize && (!deviceAlloc || m_deviceParams.size == newSize)) { return; }

				// TODO: Currently we don't reduce the capacity of vectors whose sizes are reduced. Should we?
				if (newSize > m_hostParams.capacity || (deviceAlloc && newSize > m_deviceParams.capacity))
				{
					// Mimic std::vector by growing the capacity in powers of 1.5
					const int newCapacity = max(newSize, uint(std::pow(1.5f, std::ceil(std::log(float(newSize)) / std::log(1.5f)))));
					ReserveImpl(newCapacity, deviceAlloc, deviceCopy);
				}

				m_hostParams.size = newSize;
				if (deviceAlloc || (m_hostParams.flags & kVectorUnifiedMemory))
				{
					m_deviceParams.size = newSize;
				}

				//Log::System("Size of vector '%s' changed to %i.", GetAssetID().c_str(), m_hostParams.size);
			}
		};

		// Wrapper class that hides the messy template parameterisation
		template<typename ElementType>
		class Vector : public Host::VectorBase < ElementType, Host::Vector<ElementType>, Device::Vector<ElementType>>
		{
			using Super = VectorBase < ElementType, Host::Vector<ElementType>, Device::Vector<ElementType>>;
		public:
			__host__ Vector(const std::string& id, const uint flags, cudaStream_t hostStream) : Super(id, flags, hostStream) {}
			__host__ Vector(const std::string& id, const uint size, const uint flags, cudaStream_t hostStream) : Super(id, size, flags, hostStream) {}
			__host__ virtual ~Vector() { OnDestroyAsset(); }
		};
	}	

	template<typename ElementType, typename HostType, typename DeviceType>
	__global__ void KernelFill(Device::VectorBase<ElementType, HostType, DeviceType>* cu_vector, const ElementType value)
	{
		if (kKernelIdx < cu_vector->Size())
		{
			cu_vector->Fill(kKernelIdx, value);
		}
	}
}