/*
*  DeviceObject retains ownership of device objects pointers and automatically upcasts them if required. 
* 
* */

#pragma once

#include "CudaCommonIncludes.cuh"
#include "CudaAsset.cuh"
#include "generic/Log.h"
#include <typeinfo>
#include <unordered_map>
#include <type_traits>

template<typename ObjectType, typename... Pack>
__global__ void KernelDeviceAllocCreateDeviceInstance(ObjectType** newInstance, Pack... args)
{
	assert(newInstance);
	assert(!*newInstance);

	*newInstance = new ObjectType(args...);

	assert(*newInstance);
}

template<typename ObjectType>
__global__ void KernelDeviceAllocDestroyDeviceInstance(ObjectType* cu_instance)
{
	if (cu_instance != nullptr) { delete cu_instance; }
}

template<typename ObjectType, typename CastType>
__global__ void KernelDeviceAllocStaticCastOnDevice(ObjectType** inputPtr, CastType** outputPtr)
{
	assert(inputPtr);
	assert(outputPtr);
	assert(*inputPtr);

	*outputPtr = static_cast<CastType*>(*inputPtr);
}

namespace Cuda
{
	template<typename SuperType, typename = std::enable_if<std::is_base_of<Device::Asset, SuperType>::value>>
    class DeviceObject
    {
		template<typename ObjectType, typename... Pack> friend inline DeviceObject<ObjectType> CreateDeviceObject(Pack...);

	private:
		SuperType*							cu_superPtr;
		std::unordered_map<size_t, void*>	m_subPtrMap;

		DeviceObject(SuperType* ptr) : cu_superPtr(ptr) 
		{
			Log::System("Instantiated device object of type %s at 0x%x\n", typeid(SuperType).name(), cu_superPtr);
		}

    public:
		DeviceObject() : cu_superPtr(nullptr) {}
		DeviceObject(const DeviceObject&) = delete;
		DeviceObject(DeviceObject&& other)
		{
			cu_superPtr = other.cu_superPtr;
			other.cu_superPtr = nullptr;
			m_subPtrMap = std::move(other.m_subPtrMap);
		}

		DeviceObject operator=(DeviceObject&& rhs)
		{
			cu_superPtr = rhs.cu_superPtr;
			m_subPtrMap = std::move(rhs.m_subPtrMap);
		}

		~DeviceObject()
		{
			DestroyObject();
		}

		void DestroyObject()
		{
			if (!cu_superPtr) { return; }

			KernelDeviceAllocDestroyDeviceInstance << <1, 1 >> > (cu_superPtr);
			IsOk(cudaDeviceSynchronize());

			//GlobalResourceRegistry::Get().DeregisterDeviceMemory(GetAssetID(), sizeof(ObjectType));

			Log::System("Destroyed object of type %s at address 0x%x.", typeid(SuperType).name(), cu_superPtr);
			cu_superPtr = nullptr;
			m_subPtrMap.clear();
		}

		template<typename CastType>
		explicit operator CastType* ()
		{
			static_assert(std::is_base_of_v<CastType, SuperType>, "DeviceObject cannot be cast to an object that is not a base class of its ownership type.");
			static_assert(std::is_convertible<SuperType, CastType>::value, "Can't upcast between these inputs.");

			// If this object has already been upcasted and cached, return it here
			const size_t typeHash = typeid(CastType).hash_code();
			auto it = m_subPtrMap.find(typeHash);
			if (it != m_subPtrMap.end())
			{
				Log::System("Using cached cast from %s to %s.", typeid(SuperType).name(), typeid(CastType).name());
				return reinterpret_cast<CastType*>(it->second);
			}

			SuperType** cu_inputPtr;
			IsOk(cudaMalloc((void***)&cu_inputPtr, sizeof(SuperType)));
			IsOk(cudaMemcpy(cu_inputPtr, &cu_superPtr, sizeof(SuperType), cudaMemcpyHostToDevice));
			CastType** cu_outputPtr;
			IsOk(cudaMalloc((void***)&cu_outputPtr, sizeof(CastType*)));
			IsOk(cudaMemset(cu_outputPtr, 0, sizeof(CastType*)));

			KernelDeviceAllocStaticCastOnDevice << <1, 1 >> > (cu_inputPtr, cu_outputPtr);
			IsOk(cudaDeviceSynchronize());

			CastType* outputPtr = nullptr;
			IsOk(cudaMemcpy(&outputPtr, cu_outputPtr, sizeof(CastType*), cudaMemcpyDeviceToHost));
			IsOk(cudaFree(cu_inputPtr));
			IsOk(cudaFree(cu_outputPtr));

			Log::System("Explicitly upcast device object from %s to %s.", typeid(SuperType).name(), typeid(CastType).name());

			// Emplace the upcasted pointer in the map
			m_subPtrMap.emplace(typeHash, (void*)(outputPtr));
			return outputPtr;
		}

		explicit operator SuperType* () { return cu_superPtr; }
		SuperType* operator*() { return cu_superPtr; }
	};

	template<typename ObjectType, typename... Pack>
	__host__ inline DeviceObject<ObjectType> CreateDeviceObject(Pack... args)
	{
		static_assert(std::is_base_of<Device::Asset, ObjectType>::value, "Object must inherit Device::Asset.");
		
		ObjectType** cu_tempBuffer;
		IsOk(cudaMalloc((void***)&cu_tempBuffer, sizeof(ObjectType*)));
		IsOk(cudaMemset(cu_tempBuffer, 0, sizeof(ObjectType*)));

		KernelDeviceAllocCreateDeviceInstance<ObjectType> << <1, 1 >> > (cu_tempBuffer, args...);
		IsOk(cudaDeviceSynchronize());

		ObjectType* cu_data = nullptr;
		IsOk(cudaMemcpy(&cu_data, cu_tempBuffer, sizeof(ObjectType*), cudaMemcpyDeviceToHost));
		IsOk(cudaFree(cu_tempBuffer));

		//GlobalResourceRegistry::Get().RegisterDeviceMemory(GetAssetID(), sizeof(ObjectType));		
		return DeviceObject<ObjectType>(cu_data);
	}
}