#pragma once

#include <cuda_runtime.h>
#include "cuda_fp16.h"

#include "Assert.h"
#include "CudaAsset.cuh"
#include "generic/StdIncludes.h"
#include "thirdparty/nvidia/helper_cuda.h"
#include "generic/Constants.h"

//#define CUDA_DEVICE_GLOBAL_ASSERTS

#if defined(CUDA_DEVICE_ASSERTS) || defined(CUDA_DEVICE_GLOBAL_ASSERTS)

#define CudaDeviceAssert(condition)\
        if(!(condition)) {  \
            printf("Cuda assert: %s (file %s, line %d).\n", #condition __FILE__, __LINE__);  \
            assert(condition);  \
        }

#define CudaDeviceAssertMsg(condition, message) \
        if(!(condition)) {  \
            printf("Cuda assert: %s (file %s, line %d).\n", message, __FILE__, __LINE__);  \
            assert(condition);  \
        }

#define CudaDeviceAssertFmt(condition, message, ...) \
        if(!(condition)) {  \
			printf("Cuda assert: "); \
			printf(message, __VA_ARGS__); \
            printf(" (file %s, line %d)\n", __FILE__, __LINE__);  \
            assert(condition);  \
        }

#else
	#define CudaDeviceAssert(condition)
	#define CudaDeviceAssertMsg(condition, message) 
	#define CudaDeviceAssertFmt(condition, message, ...)
#endif

template <typename T>
__host__ inline void CudaHostAssert(T result, char const* const func, const char* const file, const int line)
{
	if (result != 0)
	{
		AssertMsgFmt(false,
			"(file %s, line %d) CUDA returned error code=%d(%s) \"%s\" \n",
			file, line, (unsigned int)result, _cudaGetErrorEnum(result), func);
	}
}

#define IsOk(val) CudaHostAssert((val), #val, __FILE__, __LINE__)

#define CudaPrintVar(var, kind) printf(#var ": %" #kind "\n", var)

namespace Cuda
{		
	enum MemoryAllocFlags : uint { kCudaMemoryManaged = 1 };
	
	template<typename ObjectType>
	__host__ inline void GuardedFreeDeviceArray(const std::string& assetId, const size_t numElements, ObjectType** deviceData)
	{
		Assert(deviceData);
		if (*deviceData != nullptr)
		{
			IsOk(cudaFree(*deviceData));
			*deviceData = nullptr;

			GlobalResourceRegistry::Get().DeregisterDeviceMemory(assetId, sizeof(ObjectType) * numElements);
		}
	} 

	template<typename ObjectType>
	__host__ inline void GuardedFreeDeviceObject(const std::string& assetId, ObjectType** deviceData)
	{
		GuardedFreeDeviceArray(assetId, 1, deviceData);
	}

	template<typename ObjectType>
	__host__ inline void GuardedAllocDeviceArray(const std::string& assetId, const size_t numElements, ObjectType** deviceObject, const uint flags = 0)
	{
		Assert(deviceObject);
		AssertMsg(*deviceObject == nullptr, "Memory is already allocated.");

		if (numElements == 0) { return; }

		const size_t arraySize = sizeof(ObjectType) * numElements;

		if (flags & kCudaMemoryManaged)
		{
			IsOk(cudaMalloc((void**)deviceObject, arraySize));
		}
		else
		{
			IsOk(cudaMalloc((void**)deviceObject, arraySize));
		}

		GlobalResourceRegistry::Get().RegisterDeviceMemory(assetId, arraySize);
	}
	 
	template<typename ObjectType>
	__host__ inline void GuardedAllocDeviceObject(const std::string& assetId, ObjectType** deviceObject, const uint flags = 0)
	{
		GuardedAllocDeviceArray(assetId, 1, deviceObject, flags);
	}

	template<typename ObjectType>
	__host__ inline void GuardedAllocAndCopyToDeviceArray(const std::string& assetId, ObjectType** deviceObject, size_t numElements, ObjectType* hostData, const uint flags = 0)
	{
		Assert(hostData);

		GuardedAllocDeviceArray(assetId, numElements, deviceObject, flags);

		IsOk(cudaMemcpy(*deviceObject, hostData, sizeof(ObjectType) * numElements, cudaMemcpyHostToDevice));
	}

	template<typename ObjectType, typename... Pack>
	__global__ void KernelCreateDeviceInstance(ObjectType** newInstance, Pack... args)
	{
		assert(newInstance);
		assert(!*newInstance);

		*newInstance = new ObjectType(args...);

		assert(*newInstance);
	}

	template<typename ObjectType, typename... Pack>
	__host__ inline ObjectType* InstantiateOnDevice(const std::string& assetId, Pack... args)
	{
		ObjectType** cu_tempBuffer;
		IsOk(cudaMalloc((void***)&cu_tempBuffer, sizeof(ObjectType*)));
		IsOk(cudaMemset(cu_tempBuffer, 0, sizeof(ObjectType*)));

		KernelCreateDeviceInstance << <1, 1 >> > (cu_tempBuffer, args...);
		IsOk(cudaDeviceSynchronize());

		ObjectType* cu_data = nullptr;
		IsOk(cudaMemcpy(&cu_data, cu_tempBuffer, sizeof(ObjectType*), cudaMemcpyDeviceToHost));
		IsOk(cudaFree(cu_tempBuffer));

		GlobalResourceRegistry::Get().RegisterDeviceMemory(assetId, sizeof(ObjectType));

		Log::System("Instantiated device object at 0x%x\n", cu_data);
		return cu_data;
	}

	// Instantiate an instance of ObjectType, copies params to device memory, and passes it with the ctor parameters
	template<typename ObjectType, typename ParamsType, typename... Pack>
	__host__ inline ObjectType* InstantiateOnDeviceWithParams(const std::string& assetId, const ParamsType& params, Pack... args)
	{
		static_assert(std::is_standard_layout<ParamsType>::value, "Parameter structure must be standard layout type.");
		
		ParamsType* cu_params;
		IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
		IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

		ObjectType* cu_data = InstantiateOnDevice<ObjectType>(cu_params, args...);

		IsOk(cudaFree(cu_params));		
		return cu_data;
	}	

	template<typename ObjectType>
	__host__ bool IsParameterStandardType(const ObjectType t)
	{ 
		return std::is_standard_layout<ObjectType>::value;
	}

	template<typename ObjectType, typename... Pack>
	__host__ bool IsParameterStandardType(const ObjectType t, const Pack... pack)
	{
		if (!IsParameterStandardType(t)) { return false; }

		return IsParameterStandardType(pack...);
	}

	template<typename ObjectType, typename... ParameterPack>
	__global__ static void KernelSynchronise(ObjectType* cu_object, ParameterPack... pack)
	{
		assert(cu_object);
		cu_object->Synchronise(pack...);
	}

	template<typename ObjectType, typename... ParameterPack>
	__host__ static void Synchronise(ObjectType* cu_object, ParameterPack... pack)
	{	
		Assert(cu_object);
		Assert(IsParameterStandardType(pack...));
		
		KernelSynchronise << <1, 1 >> > (cu_object, pack...);
		IsOk(cudaDeviceSynchronize());
	}

	template<typename ObjectType, typename ParamsType>
	__global__ static void KernelSynchroniseObjects(ObjectType* cu_object, const size_t hostParamsSize, const ParamsType* cu_params)
	{
		// Check that the size of the object in the device matches that of the host. Empty base optimisation can bite us here. 
		assert(cu_object);
		assert(cu_params);
		assert(sizeof(ParamsType) == hostParamsSize);

		cu_object->Synchronise(*cu_params);
	}

	template<typename ObjectType, typename ParamsType>
	__host__ static void SynchroniseObjects(ObjectType* cu_object, const ParamsType& params)
	{
		Assert(cu_object);
		AssertMsg(std::is_standard_layout<ParamsType>::value, "Object structure must be standard layout type.");

		ParamsType* cu_params;
		IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
		IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

		IsOk(cudaDeviceSynchronize());
		KernelSynchroniseObjects << <1, 1 >> > (cu_object, sizeof(ParamsType), cu_params);
		IsOk(cudaDeviceSynchronize());

		IsOk(cudaFree(cu_params));
	}

	template<typename ObjectType>
	__global__ inline void KernelDestroyDeviceInstance(ObjectType* cu_instance)
	{
		if (cu_instance != nullptr) { delete cu_instance; }
	}

	template<typename ObjectType>
	__host__ static void DestroyOnDevice(const std::string& assetId, ObjectType*& cu_data)
	{
		if (cu_data == nullptr) { return; }
		
		KernelDestroyDeviceInstance << <1, 1 >> >(cu_data);		
		IsOk(cudaDeviceSynchronize());

		GlobalResourceRegistry::Get().DeregisterDeviceMemory(assetId, sizeof(ObjectType));

		cu_data = nullptr;
	}

	// Defines a generic kernel function that invokes the method in the referenced class
#define DEFINE_KERNEL_PASSTHROUGH_ARGS(FunctionName) \
        template<typename ObjectType, typename... Pack>\
        __global__ void Kernel##FunctionName(ObjectType* object, Pack... pack) \
        { \
            assert(object); \
            object->FunctionName(pack...); \
        }

#define DEFINE_KERNEL_PASSTHROUGH(FunctionName) \
        template<typename ObjectType>\
        __global__ void Kernel##FunctionName(ObjectType* object) \
        { \
            assert(object); \
            object->FunctionName(); \
        }
}