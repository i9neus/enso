#pragma once

#include "math/CudaMath.cuh"
#include "CudaCommonIncludes.cuh"

namespace Cuda
{
	// Substitute for std::is_base_of where base class is templatised
	// https://stackoverflow.com/questions/34672441/stdis-base-of-for-template-classes
	template < template <typename...> class base, typename derived>
	struct is_base_of_template_impl
	{
		template<typename... Ts>
		static constexpr std::true_type  test(const base<Ts...>*);
		static constexpr std::false_type test(...);
		using type = decltype(test(std::declval<derived*>()));
	};

	template < template <typename...> class base, typename derived>
	using is_base_of_template = typename is_base_of_template_impl<base, derived>::type;
	
	//enum class AccessSignal : uint { kUnlocked, kReadLocked, kWriteLocked };

	enum CudaVectorFlags : uint
	{
		kVectorUnifiedMemory = 1,
		kVectorHostAlloc = 2,
		kVectorPersistentSyncData = 4,
		kVectorSyncUpload = 8,
		kVectorSyncDownload = 16
	};
	
	namespace Host
	{
		// Forward declare some things we'll need for friendship and tagging
		//template<typename T, typename S> class Vector;
	}

	struct VectorParams
	{
		uint size = 0;
		uint capacity = 0;
		uint flags = 0;
	};

	template<typename Type>
	class VectorInterface
	{
	public:
		__host__ __device__ VectorInterface() :
			m_localData(nullptr) {}
		__host__ __device__ ~VectorInterface() {}

		__host__ __device__ __forceinline__ unsigned int		Size() const { return m_localParams.size; }
		__host__ __device__ __forceinline__ unsigned int		Capacity() const { return m_localParams.capacity; }
		__host__ __device__ __forceinline__ bool				IsEmpty() const { return m_localParams.size == 0; }
		__host__ __device__ __forceinline__ unsigned int		MemorySize() const { return m_localParams.size * sizeof(T); }

		__host__ __device__ Type* Data() { return m_localData; }
		__host__ __device__ Type& operator[](const uint idx) 
		{ 
			assert(idx < m_localParams.size);
			return m_localData[idx]; 
		}
		__host__ __device__ const Type& operator[](const uint idx) const 
		{ 
			assert(idx < m_localParams.size);
			return m_localData[idx]; 
		}
		
	protected:
		Type*			m_localData;
		VectorParams	m_localParams;
	};

	namespace Device
	{
		template<typename Type>
		class Vector : public VectorInterface<Type>,
					   public Device::Asset
		{	
		public:
			__device__ Vector() {}
			__device__ ~Vector() {}

			__device__ void Synchronise(Type* data, const VectorParams& params)
			{
				m_localData = data;
				m_localParams = params;
			}
		};
	}

	namespace Host
	{
		template<typename HostType, typename DeviceType>
		class VectorBase : public VectorInterface<HostType>, 
						   public Host::Asset
		{
		protected:
			Device::Vector<DeviceType>* cu_deviceInstance;

			DeviceType*					cu_deviceData;
			VectorParams				m_deviceParams;

		public:

			template<typename Type>
			struct IteratorBase
			{
				IteratorBase(const uint idx, HostType* data) : m_idx(idx), m_data(data) {}

				IteratorBase& operator++() { ++m_idx; return *this; }
				bool operator!=(const IteratorBase& other) const { return m_idx != other.m_idx; }

				HostType& operator*() { return m_data[m_idx]; }
				HostType& operator*() const { return m_data[m_idx]; }
				HostType* operator->() { return &m_data[m_idx]; }
				HostType* operator->() const { return &m_data[m_idx]; }

			private:
				HostType* m_data;
				uint		m_idx;
			};

			using Iterator = IteratorBase<HostType>;
			using ConstIterator = IteratorBase<const HostType>;

			/*struct Iterator : IteratorBase<Iterator>
			{
				HostType& operator*() { return m_data[m_idx]; }
				Iterator(const uint idx, HostType* data) : IteratorBase<Iterator>(idx, data) {}
			};
			struct ConstIterator : IteratorBase<ConstIterator>
			{
				const HostType& operator*() const { return m_data[m_idx]; }
				ConstIterator(const uint idx, HostType* data) : IteratorBase<ConstIterator>(idx, data) {}
			};*/

		public:
			__host__ VectorBase(const std::string& id, const uint flags, cudaStream_t hostStream) :
				Asset(id),
				cu_deviceInstance(nullptr),
				cu_deviceData(nullptr)
			{
				m_hostStream = hostStream;
				m_localParams.flags = flags;

				static_assert(std::is_trivial<HostType>::value || std::is_move_assignable<HostType>::value,
					"HostType is neither trivial not move-assignable");

				AssertMsg(!((m_localParams.flags & kVectorUnifiedMemory) && !(m_localParams.flags & kVectorHostAlloc)),
					"Must specify kVectorHostAlloc when using kVectorUnifiedMemory.");

				//AssertMsg(std::is_same<HostType, DeviceType>::value || !(m_localParams.flags & kVectorUnifiedMemory),
				//	"kVectorUnifiedMemory can only be used when host and device types are the same.");

				// Create a device instance
				//cu_deviceInstance = InstantiateOnDevice<Device::Vector<DeviceType>>(id);
			}

			__host__ VectorBase(const std::string& id, const uint size, const uint flags, cudaStream_t hostStream) :
				VectorBase(id, flags, hostStream)
			{
				// Allocate and sync the memory
				ResizeImpl(size, true, false);
			}

			__host__ __inline__ VectorBase(const VectorBase& other) { operator=(other); }
			__host__ __inline__ VectorBase(VectorBase&& other) { operator=(other); }

			__host__  ~VectorBase()
			{
				OnDestroyAsset();
			}

			__host__ __inline__ VectorBase& operator=(const std::vector<HostType>& rhs)
			{	
				CopyImpl(rhs.data(), rhs.size());	
				return *this;
			}

			__host__ __inline__ VectorBase& operator=(const VectorBase& rhs)
			{
				CopyImpl(rhs.m_localData, rhs.Size());
				return *this;
			}

			__host__ VectorBase& operator=(VectorBase&& rhs)
			{
				m_localParams = rhs.m_localParams;
				m_deviceParams = rhs.m_deviceParams;
				m_localData = rhs.m_localData;
				cu_deviceData = rhs.cu_deviceData;
				m_hostStream = rhs.m_hostStream;

				rhs.Invalidate();				
			}

			__host__  virtual void OnDestroyAsset() override final
			{
				Log::Error("Destroying %s", GetAssetID());

				// Clean up device memory
				GuardedFreeDeviceArray(GetAssetID(), m_deviceParams.capacity, &cu_deviceData);
				m_deviceParams.size = 0;
				m_deviceParams.capacity = 0;

				Assert(!m_localData || m_localParams.capacity); // Sanity check
				
				// Clean up host memory (if not unified)
				if (m_localData && m_localParams.flags & kVectorHostAlloc && !(m_localParams.flags & kVectorUnifiedMemory))
				{
					if (!std::is_trivial<HostType>::value)
					{
						for (int idx = 0; idx < m_localParams.size; ++idx)
						{
							m_localData[idx].~HostType(); // In-place dtor
						}
					}

					std::free(m_localData);
					m_localData = nullptr;
					m_localParams.size = 0;
					m_localParams.capacity = 0;
				}

				// Destroy the device instance
				DestroyOnDevice(GetAssetID(), cu_deviceInstance);				
			}

			__host__ inline void Prepare()
			{
				// FIXME: On Windows, CUDA requires that cudaDeviceSynchronize() is called after kernels access shared memory. 
				if (m_localParams.flags & kVectorUnifiedMemory)
				{
					IsOk(cudaDeviceSynchronize());
				}
			}

			// Make container compatible with range-based loops
			__host__ inline Iterator		begin() { return Iterator(0, m_localData); }
			__host__ inline Iterator		end() { return Iterator(m_localParams.size, m_localData); }
			__host__ inline ConstIterator	begin() const { return ConstIterator(0, m_localData); }
			__host__ inline ConstIterator	end() const { return ConstIterator(m_localParams.size, m_localData); }

			__host__ inline Device::Vector<DeviceType>* GetDeviceInstance() 
			{ 
				// Lazily initialise the device instance so we can use this class as an ordinary host vector without additional overhead
				if (cu_deviceInstance == nullptr)
				{
					cu_deviceInstance = InstantiateOnDevice<Device::Vector<DeviceType>>(GetAssetID());
				}

				return cu_deviceInstance; 
			}

			__host__ inline HostType* GetHostData()
			{
				AssertMsgFmt(m_localParams.flags & kVectorHostAlloc, "Vector '%s' does not have host allocation.", GetAssetID().c_str());
				return m_localData;
			}

			__host__ inline uint			Size() const { return m_localParams.size; }
			__host__ inline uint			Capacity() const { return m_localParams.capacity; }
			__host__ inline bool			IsEmpty() const { return m_localParams.size == 0; }

			__host__ inline const HostType& operator[](const uint idx) const
			{
				return const_cast<VectorBase*>(this)->operator[](idx);
			}

			__host__ HostType& operator[](const uint idx)
			{
				AssertMsgFmt(m_localParams.flags & kVectorHostAlloc, "Vector '%s': trying to use [] on array that does not have host allocation.", GetAssetID().c_str());
				Assert(idx < m_localParams.size);

				return m_localData[idx];
			}

			// ---------------------------------------------------------------------------------------------------------------------------------------------	

			__host__ inline void Reserve(const uint newCapacity) { ReserveImpl(newCapacity, false, true); }
			__host__ inline void Resize(const uint newSize) { ResizeImpl(newSize, false, true); }

			__host__ inline void Grow(const uint inc) { ResizeImpl(m_localParams.size + inc, false, true); }
			__host__ inline void Shrink(const uint dec) { ResizeImpl(m_localParams.size - dec, false, true); }

			__host__ void PushBack(const HostType& element)
			{
				AssertMsg(m_localParams.flags & kVectorHostAlloc, "Calling PushBack() on a Vector that does not have host allocation.");
				Resize(m_localParams.size + 1);
				m_localData[m_localParams.size - 1] = element;
			}

			template<typename... Pack>
			__host__ void EmplaceBack(Pack... pack)
			{
				AssertMsg(m_localParams.flags & kVectorHostAlloc, "Calling EmplaceBack() on a Vector that does not have host allocation.");
				Resize(m_localParams.size + 1);
				new (&m_localData[m_localParams.size - 1]) HostType(pack...);
			}

			__host__ void PopBack()
			{
				AssertMsg(m_localParams.flags & kVectorHostAlloc, "Calling PopBack() on a Vector that does not have host allocation.");
				Resize(m_localParams.size - 1);
			}

			__host__ inline HostType& Back()
			{
				AssertMsg(m_localParams.flags & kVectorHostAlloc, "Calling Back() on a Vector that does not have host allocation.");
				return m_localData[m_localParams.size - 1];
			}

			__host__ inline HostType& Front()
			{
				AssertMsg(m_localParams.flags & kVectorHostAlloc, "Calling Front() on a Vector that does not have host allocation.");
				return m_localData[0];
			}

			__host__ inline void Clear()
			{
				Resize(0);
			}

			// Erases the contents of the vector with the specified value
			__host__ void Fill(const HostType& value)
			{
				/*Assert(cu_deviceInstance);

				constexpr int blockSize = 16 * 16;
				int gridSize = (m_deviceParams.size + (blockSize - 1)) / (blockSize);
				KernelFill << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance, value);*/

				// If the vector has host allocation, reset that too
				if (m_localParams.flags & kVectorHostAlloc && !(m_localParams.flags & kVectorUnifiedMemory))
				{
					for (int idx = 0; idx < m_localParams.size; ++idx)
					{
						m_localData[idx] = value;
					}
				}

				IsOk(cudaDeviceSynchronize());
			}

			// Zeroes the contents of the vector
			__host__ void Wipe()
			{
				/*Assert(cu_deviceInstance);
				IsOk(cudaMemset(m_localData, 0, sizeof(HostType) * m_localParams.size));
				IsOk(cudaStreamSynchronize(m_hostStream));*/

				// If the vector has host allocation, reset that too
				if (m_localParams.flags & kVectorHostAlloc && !(m_localParams.flags & kVectorUnifiedMemory))
				{
					std::memset(m_localData, 0, sizeof(HostType) * m_localParams.size);
				}
			}

		protected:
			__host__ void CopyImpl(HostType* data, const uint newSize)
			{
				AssertMsg(m_localParams.flags & kVectorHostAlloc, "Must specify kVectorHostAlloc to use operator=");

				Resize(newSize);
				
				if (std::is_trivially_copyable<HostType>::value)
				{
					std::memcpy(m_localData, data, sizeof(HostType) * m_localParams.size);
				}
				else
				{
					for (int idx = 0; idx < m_localParams.size; ++idx)
					{
						data[idx] = data[idx];
					}
				}
			}

			__host__ void Invalidate()
			{
				m_localParams.size = 0;
				m_localParams.capacity = 0;
				m_localData = nullptr;
				cu_deviceData = nullptr;
			}

			template<bool CopyPtr>
			__host__ __inline__ void CopyDeviceToHostPtr() {}			

			template<>
			__host__ __inline__ void CopyDeviceToHostPtr<true>()
			{
				m_localData = cu_deviceData;
			}

			__host__ void ReserveImpl(const uint newCapacity, const bool deviceAlloc, const bool deviceCopy)
			{
				// If the device is being explicitly synced or we're using unified memory, realloc the device memory 
				if (newCapacity > m_deviceParams.capacity && (deviceAlloc || (m_localParams.flags & kVectorUnifiedMemory)))
				{
					// Don't adjust capacity if it means reducing the size of the array
					Assert(newCapacity >= m_deviceParams.size);

					// Allocate a new block of memory
					DeviceType* newDeviceData = nullptr;
					GuardedAllocDeviceArray(GetAssetID(), newCapacity, &newDeviceData, (m_localParams.flags & kVectorUnifiedMemory) ? kCudaMemoryManaged : 0u);

					// If we're syncing from host to device, don't copy the data from the old device buffer because it'll just get overwritten anyway
					if (cu_deviceData)
					{
						if (deviceCopy || (m_localParams.flags & kVectorUnifiedMemory))
						{
							IsOk(cudaMemcpy(newDeviceData, cu_deviceData, m_localParams.size * sizeof(DeviceType), cudaMemcpyDeviceToDevice));
						}

						// Deallocate the old memory
						GuardedFreeDeviceArray(GetAssetID(), m_deviceParams.capacity, &cu_deviceData);
					}

					m_deviceParams.capacity = newCapacity;
					cu_deviceData = newDeviceData;

					//Log::System("Capacity of device vector '%s' changed to %i.", GetAssetID().c_str(), m_localParams.capacity);
				}

				// Unified memory is accessible on both host and and device so just copy the pointer
				if (m_localParams.flags & kVectorUnifiedMemory)
				{
					CopyDeviceToHostPtr<std::is_same<HostType, DeviceType>::value>();
					m_localParams.capacity = newCapacity;
				}

				// Otherwise, reallocate the host data
				else if (m_localParams.flags & kVectorHostAlloc && newCapacity > m_localParams.capacity)
				{
					// Don't adjust capacity if it means reducing the size of the array
					Assert(newCapacity >= m_localParams.size);

					HostType* newHostData = static_cast<HostType*>(std::malloc(sizeof(HostType) * newCapacity));
					if (m_localData)
					{
						if (std::is_trivial<HostType>::value)
						{
							// Trivial types mean we can do a straight copy
							std::memcpy(newHostData, m_localData, m_localParams.size * sizeof(HostType));
						}
						else
						{
							// Default-initialise and move the contents of the vector into its new buffer
							for (int idx = 0; idx < m_localParams.size; ++idx)
							{
								new (&newHostData[idx]) HostType();
								newHostData[idx] = std::move(m_localData[idx]);
							}
						}
						// Deallocate the old buffer
						std::free(m_localData);
					}

					// Update the pointers and the capacity
					m_localData = newHostData;
					m_localParams.capacity = newCapacity;

					//Log::System("Capacity of host vector '%s' changed to %i.", GetAssetID().c_str(), m_localParams.capacity);
				}
			}

			__host__ void ResizeImpl(const uint newSize, const bool deviceAlloc, const bool deviceCopy)
			{
				// Don't resize if nothing has changed
				if (m_localParams.size == newSize && (!deviceAlloc || m_deviceParams.size == newSize)) { return; }

				// TODO: Currently we don't reduce the capacity of vectors whose sizes are reduced. Should we?
				if (newSize > m_localParams.capacity || (deviceAlloc && newSize > m_deviceParams.capacity))
				{
					// Mimic std::vector by growing the capacity in powers of 1.5
					const int newCapacity = max(newSize, uint(std::pow(1.5f, std::ceil(std::log(float(newSize)) / std::log(1.5f)))));
					ReserveImpl(newCapacity, deviceAlloc, deviceCopy);
				}

				// For non-trivial types, make sure objects are properly constructed and destructed
				if (!std::is_trivial<HostType>::value && m_localParams.flags & kVectorHostAlloc)
				{
					for (int idx = newSize; idx < m_localParams.size; ++idx)
					{
						m_localData[idx].~HostType(); // In-place dtor
					}
					for (int idx = m_localParams.size; idx < newSize; ++idx)
					{
						new (&m_localData[idx]) HostType(); // In-place ctor
					}
				}

				m_localParams.size = newSize;
				if (deviceAlloc || (m_localParams.flags & kVectorUnifiedMemory))
				{
					m_deviceParams.size = newSize;
				}

				//Log::System("Size of vector '%s' changed to %i.", GetAssetID().c_str(), m_localParams.size);
			}
		};

		// Generic vector where both types are the same
		template<typename CommonType>
		class Vector : public VectorBase<CommonType, CommonType>
		{
		public:
			__host__ Vector(const uint flags, cudaStream_t hostStream = nullptr) : VectorBase<CommonType, CommonType>(Asset::MakeTemporaryID(), flags, hostStream) {}
			__host__ Vector(const std::string& id, const uint flags, cudaStream_t hostStream = nullptr) : VectorBase<CommonType, CommonType>(id, flags, hostStream) {}
			__host__ Vector(const std::string& id, const uint size, const uint flags, cudaStream_t hostStream = nullptr) : VectorBase<CommonType, CommonType>(id, size, flags, hostStream) {}
			__host__ ~Vector() {};

			// Transfers the contents of the buffers between the host and device 
			__host__ void Synchronise(const uint syncFlags)
			{
				AssertMsg(!(m_localParams.flags & kVectorUnifiedMemory), "Calling Upload() on a Vector with kVectorUnifiedMemory flag set.");
				AssertMsg(m_localParams.flags & kVectorHostAlloc, "Trying to synchronise a Vector that does not have host allocation.");
				AssertMsg(syncFlags & (kVectorSyncUpload | kVectorSyncDownload), "Invalid syncronisation flags.");

				if (syncFlags == kVectorSyncUpload)
				{
					// Make sure the device data matches the host size
					if (m_localParams.size != m_deviceParams.size)
					{
						ResizeImpl(m_localParams.size, true, false);
					}

					if (cu_deviceData)
					{
						IsOk(cudaMemcpy(cu_deviceData, m_localData, sizeof(CommonType) * m_localParams.size, cudaMemcpyHostToDevice));
						Cuda::Synchronise(GetDeviceInstance(), cu_deviceData, m_deviceParams);
					}
				}
				else
				{
					if (cu_deviceData)
					{
						IsOk(cudaMemcpy(m_localData, cu_deviceData, sizeof(CommonType) * m_localParams.size, cudaMemcpyDeviceToHost));
					}
				}
			}
		};

		// Requires that HostType inherit AssetTags
		template<typename HostType>
		class AssetVector : public VectorBase<AssetHandle<typename HostType>, typename HostType::DeviceVariant*>
		{
		public:
			using DeviceType = typename HostType::DeviceVariant*;

		private:
			DeviceType*		m_deviceSyncData;

		public:
			__host__ AssetVector(const std::string& id, const uint flags, cudaStream_t hostStream) : VectorBase<HostType, DeviceType>(id, flags, hostStream)
			{
				//static_assert(is_base_of_template<AssetTags, HostType>, "AssetVector type must inherit AssetTags");
				static_assert(std::is_trivial<DeviceType>::value, "Sanity check failed. Device type is non-trivial.");
			}
			__host__ AssetVector(const std::string& id, const uint size, const uint flags, cudaStream_t hostStream) : VectorBase<HostType, DeviceType>(id, size, flags, hostStream) {}

			__host__ void Synchronise(const uint syncFlags)
			{
				AssertMsg(syncFlags == kVectorSyncUpload, "Mapped vectors only supports uploading for now.");
				AssertMsg(m_localParams.flags & kVectorHostAlloc, "Trying to synchronise a Vector that does not have host allocation.");

				// Make sure the device data matches the host size
				if (m_localParams.size != m_deviceParams.size)
				{
					ResizeImpl(m_localParams.size, true, false);
				}

				if (!cu_deviceData) { return; }

				// Allocate some temporary storage
				m_deviceSyncData = new DeviceType[m_localParams.size];
				Assert(m_deviceSyncData);

				// Convert between the host and device datatypes
				for (int idx = 0; idx < m_localParams.size; ++idx)
				{
					//m_deviceSyncData[idx] = convertFunctor(m_localData[idx]);
					m_deviceSyncData[idx] = m_localData[idx]->GetDeviceInstance();
				}

				// Copy to the device
				IsOk(cudaMemcpy(cu_deviceData, m_deviceSyncData, sizeof(DeviceType) * m_localParams.size, cudaMemcpyHostToDevice));
				delete[] m_deviceSyncData;

				// Synchronise the device data pointers and the params
				Cuda::Synchronise(GetDeviceInstance(), cu_deviceData, m_deviceParams);
			}
		};
	}

	template<typename DeviceType>
	__global__ void KernelFill(Device::Vector<DeviceType>* cu_vector, const DeviceType value)
	{
		if (kKernelIdx < cu_vector->Size())
		{
			cu_vector->Fill(kKernelIdx, value);
		}
	}
}