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
			"CUDA returned error code=%d(%s) \"%s\" \n",
			file, line, (unsigned int)result, _cudaGetErrorEnum(result), func);
	}
}

#define IsOk(val) CudaHostAssert((val), #val, __FILE__, __LINE__)

#define CudaPrintVar(var, kind) printf(#var ": %" #kind "\n", var)

namespace Cuda
{		
	template<typename T>
	__host__ inline void SafeFreeDeviceMemory(T** deviceData)
	{
		Assert(deviceData);
		if (*deviceData != nullptr)
		{
			IsOk(cudaFree(*deviceData));
			*deviceData = nullptr;
		}
	}

	template<typename T>
	__host__ inline void SafeAllocDeviceMemory(T** deviceObject, size_t numElements)
	{
		Assert(deviceObject);
		AssertMsg(*deviceObject == nullptr, "Memory is already allocated.");

		if (numElements == 0) { return; }

		const size_t arraySize = sizeof(T) * numElements;

		Log::System("Allocated %i bytes of GPU memory (%i elements)\n", arraySize, numElements);

		IsOk(cudaMalloc((void**)deviceObject, arraySize));
	}

	template<typename T>
	__host__ inline void SafeAllocAndCopyToDeviceMemory(T** deviceObject, size_t numElements, T* hostData)
	{
		Assert(hostData);
		
		SafeAllocDeviceMemory(deviceObject, numElements);

		IsOk(cudaMemcpy(*deviceObject, hostData, sizeof(T) * numElements, cudaMemcpyHostToDevice));
	}

	template<typename T, typename... Pack>
	__global__ inline void KernelCreateDeviceInstance(T** newInstance, Pack... args)
	{
		assert(newInstance);
		assert(!*newInstance);
		
		*newInstance = new T(args...); 
	}

	template<typename ObjectType, typename... Pack>
	__host__ inline ObjectType* InstantiateOnDevice(Pack... args)
	{		
		ObjectType** cu_tempBuffer;
		IsOk(cudaMalloc((void***)&cu_tempBuffer, sizeof(ObjectType*)));
		IsOk(cudaMemset(cu_tempBuffer, 0, sizeof(ObjectType*)));

		KernelCreateDeviceInstance<<<1, 1>>>(cu_tempBuffer, args...);
		IsOk(cudaDeviceSynchronize());

		ObjectType* cu_data = nullptr;
		IsOk(cudaMemcpy(&cu_data, cu_tempBuffer, sizeof(ObjectType*), cudaMemcpyDeviceToHost));
		IsOk(cudaFree(cu_tempBuffer));

		Log::System("Instantiated device object at 0x%x\n", cu_data);
		return cu_data;
	}

	// Instantiate an instance of ObjectType, copies params to device memory, and passes it with the ctor parameters
	template<typename ObjectType, typename ParamsType, typename... Pack>
	__host__ inline ObjectType* InstantiateOnDeviceWithParams(const ParamsType& params, Pack... args)
	{
		static_assert(std::is_standard_layout<ParamsType>::value, "Parameter structure must be standard layout type.");
		
		ParamsType* cu_params;
		IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
		IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

		ObjectType* cu_data = InstantiateOnDevice<ObjectType>(cu_params, args...);

		IsOk(cudaFree(cu_params));
		return cu_data;
	}	

	template<typename T>
	__host__ bool IsParameterStandardType(const T t) 
	{ 
		return std::is_standard_layout<T>::value;
	}

	template<typename T, typename... Pack>
	__host__ bool IsParameterStandardType(const T t, const Pack... pack)
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

		KernelSynchroniseObjects << <1, 1 >> > (cu_object, sizeof(ParamsType), cu_params);
		IsOk(cudaDeviceSynchronize());

		IsOk(cudaFree(cu_params));
	}

	template<typename T>
	__global__ inline void KernelDestroyDeviceInstance(T* cu_instance)
	{
		if (cu_instance != nullptr) { delete cu_instance; }
	}

	template<typename T>
	__host__ static void DestroyOnDevice(T*& cu_data)
	{
		if (cu_data == nullptr) { return; }
		
		KernelDestroyDeviceInstance<<<1, 1>>>(cu_data);		
		IsOk(cudaDeviceSynchronize());

		cu_data = nullptr;
	}
}