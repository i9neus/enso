#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
	enum class AccessSignal : uint { kUnlocked, kReadLocked, kWriteLocked };
	enum class ManagedArrayLayout : uint { k1D, k2D };
	
	namespace Host
	{
		// Forward declare some things we'll need for friendship and tagging
		template<typename T, typename HostType, typename DeviceType> class ManagedArray;
		template<typename T> class Array;
	}
	
	namespace Device
	{
		template<typename T, typename HostType, typename DeviceType>
		class ManagedArray : public Device::Asset, public AssetTags<HostType, DeviceType>
		{
			friend HostType;
			template<typename A, typename B, typename C> friend class Host::ManagedArray;
		public:
			__device__ ManagedArray(const uint size, const ManagedArrayLayout& layout, T* data) :
				m_size(size), m_layout(layout), cu_data(data), m_signal(AccessSignal::kUnlocked) { }
			__device__ ~ManagedArray() = default;

			__host__ __device__ inline unsigned int Size() const { return m_size; }
			__host__ __device__ inline unsigned int MemorySize() const { return m_size * sizeof(T); }

			__device__ T* GetData() { return cu_data; }
			__device__ T& operator[](const uint idx) { return cu_data[idx]; }
			__device__ const T& operator[](const uint idx) const { return cu_data[idx]; }
			__device__ inline void Clear(const uint idx, const T& value)
			{
				if (idx < m_size) { cu_data[idx] = value; }
			}

		protected:
			__host__ __device__ ManagedArray() : m_size(0), cu_data(nullptr) {}

			uint				m_size;
			ManagedArrayLayout	m_layout;
			T*					cu_data;
			AccessSignal		m_signal;
		};

		template<typename T>
		class Array : public Device::ManagedArray<T, Host::Array<T>, Device::Array<T>>
		{
		public:
			__device__ Array(const uint size, const ManagedArrayLayout& layout, T* data) : ManagedArray<T, Host::Array<T>, Device::Array<T>>(size, layout, data) {}
		protected:
			__host__ __device__ Array() : ManagedArray<T, Host::Array<T>, Device::Array<T>>() {}
		};

		//template class Array<CompressedRay>;
	}

	namespace Host
	{
		template<typename T, typename HostType, typename DeviceType>
		class ManagedArray : public Host::Asset, public AssetTags<HostType, DeviceType>
		{
		protected:
			DeviceType*											cu_deviceData;
			Device::ManagedArray<T, HostType, DeviceType>		m_hostData;

			cudaStream_t	  m_hostStream;			
			int			      m_threadsPerBlock;
			int			      m_numBlocks;

		public:
			__host__ ManagedArray() : cu_deviceData(nullptr) { }
			__host__ ManagedArray(const uint size, cudaStream_t hostStream);
			__host__  ~ManagedArray() { OnDestroyAsset(); }

			__host__  virtual void OnDestroyAsset() override final;

			__host__ cudaStream_t GetHostStream() const { return m_hostStream; }
			__host__ DeviceType* GetDeviceInstance() const
			{
				AssertMsg(cu_deviceData, "ManagedArray has not been initialised!");
				return cu_deviceData;
			}
			__host__ inline const Device::ManagedArray<T, HostType, DeviceType>& GetHostInstance() const { return m_hostData; }
			__host__ inline bool IsCreated() const { return cu_deviceData != nullptr; }
			__host__ inline int ThreadsPerBlock() const { return m_threadsPerBlock; }
			__host__ inline int NumBlocks() const { return m_numBlocks; }

			__host__ void SignalChange(cudaStream_t hostStream, const unsigned int currentState, const unsigned int newState);
			__host__ inline void SignalSetRead(cudaStream_t hostStream = nullptr) { SignalChange(hostStream, AccessSignal::kUnlocked, AccessSignal::kReadLocked); }
			__host__ inline void SignalUnsetRead(cudaStream_t hostStream = nullptr) { SignalChange(hostStream, AccessSignal::kReadLocked, AccessSignal::kUnlocked); }
			__host__ inline void SignalSetWrite(cudaStream_t hostStream = nullptr) { SignalChange(hostStream, AccessSignal::kUnlocked, AccessSignal::kWriteLocked);	}
			__host__ inline void SignalUnsetWrite(cudaStream_t hostStream = nullptr) { SignalChange(hostStream, AccessSignal::kWriteLocked, AccessSignal::kUnlocked); }
			__host__ void Clear(const T& value);
		};

		template<typename T>
		class Array : public Host::ManagedArray<T, Host::Array<T>, Device::Array<T>>
		{
			using Super = ManagedArray<T, Host::Array<T>, Device::Array<T>>;
		public:
			__host__ Array(const uint size, cudaStream_t hostStream) : Super(size, hostStream) {}
			__host__ virtual ~Array() { OnDestroyAsset(); }
		};		
	}

	template<typename T>
	__global__ void KernelSignalChange(T* array, const unsigned int currentState, const unsigned int newState)
	{
		atomicCAS(array->AccessSignal(), currentState, newState);
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ Host::ManagedArray<T, HostType, DeviceType>::ManagedArray(const uint size, cudaStream_t hostStream) :
		cu_deviceData(nullptr)
	{
		// Prepare the host data
		m_hostData.m_size = size;
		m_hostData.m_signal = AccessSignal::kUnlocked;
		m_hostData.m_layout = ManagedArrayLayout::k1D;

		SafeAllocDeviceMemory(&m_hostData.cu_data, size);

		cu_deviceData = InstantiateOnDevice<DeviceType>(m_hostData.m_size, m_hostData.m_layout, m_hostData.cu_data);

		m_hostStream = hostStream;
		m_threadsPerBlock = 16 * 16;
		m_numBlocks = (size + (m_threadsPerBlock - 1)) / (m_threadsPerBlock);
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ void Host::ManagedArray<T, HostType, DeviceType>::SignalChange(cudaStream_t otherStream, const unsigned int currentState, const unsigned int newState)
	{
		cudaStream_t hostStream = otherStream ? otherStream : m_hostStream;
		KernelSignalChange << < 1, 1, 0, hostStream >> > (cu_deviceData, currentState, newState);
	}

	template<typename T, typename HostType, typename DeviceType>
	__global__ void KernelClear(Device::ManagedArray<T, HostType, DeviceType>* cu_array, const T value)
	{
		//if (*(array->AccessSignal()) != kManagedArrayWriteLocked) { return; }
		cu_array->Clear(blockIdx.x * blockDim.x + threadIdx.x, value);
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ void Host::ManagedArray<T, HostType, DeviceType>::Clear(const T& value)
	{
		KernelClear << < m_numBlocks, m_threadsPerBlock, 0, m_hostStream >> > (cu_deviceData, value);
		IsOk(cudaStreamSynchronize(m_hostStream));
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ void Host::ManagedArray<T, HostType, DeviceType>::OnDestroyAsset()
	{
		DestroyOnDevice(&cu_deviceData);
		SafeFreeDeviceMemory(&m_hostData.cu_data);
	}
}