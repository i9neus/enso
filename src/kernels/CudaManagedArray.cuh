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
			__host__ ManagedArray() : cu_data(nullptr), m_size(0) {}
			__device__ ManagedArray(const ManagedArrayLayout& layout) :
				m_size(0), m_layout(layout), cu_data(nullptr), m_signal(AccessSignal::kUnlocked) { }
			__device__ ~ManagedArray() {}

			__host__ __device__ __forceinline__ unsigned int Size() const { return m_size; }
			__host__ __device__ __forceinline__ unsigned int MemorySize() const { return m_size * sizeof(T); }

			__device__ T* GetData() { return cu_data; }
			__device__ T& operator[](const uint idx) { return cu_data[idx]; }
			__device__ const T& operator[](const uint idx) const { return cu_data[idx]; }

			__device__ __forceinline__ void Clear(const uint idx, const T& value)
			{
				if (idx < m_size) { cu_data[idx] = value; }
			}
			
			__device__ void Synchronise(T* data, const uint size)
			{
				cu_data = data;
				m_size = size;
			}

		protected:
			uint				m_size;
			ManagedArrayLayout	m_layout;
			T*					cu_data;
			AccessSignal		m_signal;
		};

		template<typename T>
		class Array : public Device::ManagedArray<T, Host::Array<T>, Device::Array<T>>
		{
		public:
			__host__ __device__ Array() : ManagedArray<T, Host::Array<T>, Device::Array<T>>(ManagedArrayLayout::k1D) {}
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
			int			      m_blockSize;
			int			      m_gridSize;

		public:
			__host__ ManagedArray(cudaStream_t hostStream);
			__host__ ManagedArray(const uint size, cudaStream_t hostStream);
			__host__  virtual ~ManagedArray() { }

			__host__  virtual void OnDestroyAsset() override final;

			__host__ cudaStream_t GetHostStream() const { return m_hostStream; }
			__host__ DeviceType* GetDeviceInstance() const
			{
				AssertMsg(cu_deviceData, "ManagedArray has not been initialised!");
				return cu_deviceData;
			}
			__host__ inline const Device::ManagedArray<T, HostType, DeviceType>& GetHostInstance() const { return m_hostData; }
			__host__ inline bool IsCreated() const { return cu_deviceData != nullptr; }
			__host__ inline int GetBlockSize() const { return m_blockSize; }
			__host__ inline int GetGridSize() const { return m_gridSize; }

			__host__ inline uint Size() const { return m_hostData.m_size; }
			__host__ void Resize(const uint size);
			__host__ inline bool Expand(const uint size);
			__host__ inline bool ExpandToNearestPow2(const uint size);
			__host__ void Download(std::vector<T>& rawData) const;
			__host__ void Upload(const std::vector<T>& rawData);
			__host__ void Swap(ManagedArray& other);
			__host__ void Replace(const ManagedArray& other);

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
			__host__ Array(cudaStream_t hostStream) : Super(hostStream) {}
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
	__host__ Host::ManagedArray<T, HostType, DeviceType>::ManagedArray(cudaStream_t hostStream) :
		cu_deviceData(nullptr)
	{
		// Prepare the host data
		m_hostData.m_signal = AccessSignal::kUnlocked;
		m_hostData.m_layout = ManagedArrayLayout::k1D;

		m_hostStream = hostStream;
		m_blockSize = 16 * 16;

		// Create a device instance
		cu_deviceData = InstantiateOnDevice<DeviceType>();
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ Host::ManagedArray<T, HostType, DeviceType>::ManagedArray(const uint size, cudaStream_t hostStream) :
		Host::ManagedArray<T, HostType, DeviceType>(hostStream)
	{
		m_gridSize = (size + (m_blockSize - 1)) / (m_blockSize);

		// Allocate and sync the memory
		Resize(size);
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ void Host::ManagedArray<T, HostType, DeviceType>::Resize(const uint newSize)
	{		
		if (m_hostData.m_size == newSize) { return; }
		
		SafeFreeDeviceMemory(&m_hostData.cu_data);
		SafeAllocDeviceMemory(&m_hostData.cu_data, newSize);
		m_hostData.m_size = newSize;

		Cuda::Synchronise(static_cast<Device::ManagedArray<T, HostType, DeviceType>*>(cu_deviceData), m_hostData.cu_data, newSize); 
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ bool Host::ManagedArray<T, HostType, DeviceType>::Expand(const uint newSize)
	{
		if (m_hostData.m_size < newSize)
		{
			Resize(newSize);
			return true;
		}
		return false;
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ bool Host::ManagedArray<T, HostType, DeviceType>::ExpandToNearestPow2(const uint newSize)
	{
		if (m_hostData.m_size < newSize)
		{
			Resize(NearestPow2Ceil(newSize));
			return true;
		}
		return false;
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ void Host::ManagedArray<T, HostType, DeviceType>::Swap(ManagedArray& other)
	{
		AssertMsgFmt(other.Size() == Size(), "Array size mismatch. This array has %i elements, other is %i.", Size(), other.Size());
		
		std::swap(m_hostData.cu_data, other.m_hostData.cu_data);

		Cuda::Synchronise(static_cast<Device::ManagedArray<T, HostType, DeviceType>*>(cu_deviceData), m_hostData.cu_data, m_hostData.m_size);
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ void Host::ManagedArray<T, HostType, DeviceType>::Replace(const ManagedArray& other)
	{				
		// Resize the data if necessary
		if (Size() != other.Size()) { Resize(other.Size()); }

		// Copy over the data
		if (Size() > 0)
		{
			IsOk(cudaMemcpy(m_hostData.cu_data, other.m_hostData.cu_data, sizeof(T) * Size(), cudaMemcpyDeviceToDevice));
		}
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
		KernelClear << < m_gridSize, m_blockSize, 0, m_hostStream >> > (cu_deviceData, value);
		IsOk(cudaStreamSynchronize(m_hostStream));
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ void Host::ManagedArray<T, HostType, DeviceType>::OnDestroyAsset()
	{
		DestroyOnDevice(cu_deviceData);
		SafeFreeDeviceMemory(&m_hostData.cu_data);
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ void Host::ManagedArray<T, HostType, DeviceType>::Download(std::vector<T>& rawData) const
	{
		Assert(m_hostData.cu_data);
		if (m_hostData.m_size == 0) { return; }

		rawData.resize(m_hostData.m_size);

		IsOk(cudaMemcpy(rawData.data(), m_hostData.cu_data, sizeof(T) * m_hostData.m_size, cudaMemcpyDeviceToHost));
	}

	template<typename T, typename HostType, typename DeviceType>
	__host__ void Host::ManagedArray<T, HostType, DeviceType>::Upload(const std::vector<T>& rawData)
	{
		if (rawData.empty()) { return; }

		Resize(rawData.size());

		IsOk(cudaMemcpy(m_hostData.cu_data, rawData.data(), sizeof(T) * rawData.size(), cudaMemcpyHostToDevice));
	}
}