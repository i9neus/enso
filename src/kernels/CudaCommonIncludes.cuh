#pragma once

#include <cuda_runtime.h>
#include "cuda_fp16.h"

#include "Assert.h"
#include "CudaAsset.cuh"
#include "generic/StdIncludes.h"
#include "thirdparty/nvidia/helper_cuda.h"

namespace Cuda
{
#define IsOk checkCudaErrors
	
	using uint = unsigned int;
	using uchar = unsigned char;
	using ushort = unsigned short;

	__host__ __device__ inline float clamp(const float& v, const float& a, const float& b) { return fmaxf(a, fminf(v, b)); }
	__host__ __device__ inline float fract(const float& v) { return fmodf(v, 1.0f); }
	__host__ __device__ inline float sign(const float& v) { return copysign(1.0f, v); }
	template<typename T> __host__ inline void echo(const T& t) { std::printf("%s\n", t.format().c_str()); }

	#define KERNEL_COORDS_IVEC2 ivec2(blockIdx.x* blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)

	template<typename T>
	inline void SafeFreeDeviceMemory(T** deviceData)
	{
		Assert(deviceData);
		if (*deviceData != nullptr)
		{
			IsOk(cudaFree(*deviceData));
			*deviceData = nullptr;
		}
	}

	template<typename T>
	inline void SafeAllocDeviceMemory(T** deviceObject, size_t numElements, T* hostData = nullptr)
	{
		Assert(deviceObject);
		AssertMsg(*deviceObject == nullptr, "Memory is already allocated.");
		const size_t arraySize = sizeof(T) * numElements;

		std::printf("Allocated %i bytes of GPU memory (%i elements)\n", arraySize, numElements);

		IsOk(cudaMalloc((void**)deviceObject, arraySize));
		if (hostData)
		{
			IsOk(cudaMemcpy(deviceObject, hostData, arraySize, cudaMemcpyHostToDevice));
		}
	}

	/*template<typename T>
	inline void SafeCreateDeviceInstance(T** deviceData, const T& instance)
	{
		Assert(deviceData);
		Assert(*deviceData == nullptr);

		IsOk(cudaMalloc((void**)deviceData, sizeof(T)));
		IsOk(cudaMemcpy(*deviceData, &instance, sizeof(T), cudaMemcpyHostToDevice));
	}*/

	template<typename T, typename... Pack>
	__global__ inline void KernelCreateDeviceInstance(T** newInstance, Pack... args)
	{
		if (newInstance && !*newInstance) { *newInstance = new T(args...); }
	}

	template<typename ObjectType, typename... Pack>
	__host__ inline ObjectType* InstantiateOnDevice(Pack... args)
	{		
		ObjectType** cu_tempBuffer;
		IsOk(cudaMalloc((void***)&cu_tempBuffer, sizeof(ObjectType*)));

		KernelCreateDeviceInstance << < 1, 1 >> > (cu_tempBuffer, args...);
		IsOk(cudaDeviceSynchronize());

		ObjectType* cu_data;
		IsOk(cudaMemcpy(&cu_data, cu_tempBuffer, sizeof(ObjectType*), cudaMemcpyDeviceToHost));
		IsOk(cudaFree(cu_tempBuffer));
		return cu_data;
	}

	// Instantiate an instance of ObjectType, copies params to device memory, and passes it with the ctor parameters
	template<typename ObjectType, typename ParamsType, typename... Pack>
	__host__ inline ObjectType* InstantiateOnDeviceWithParams(const ParamsType& params, Pack... args)
	{
		ParamsType* cu_params;
		IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
		IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

		ObjectType* cu_data = InstantiateOnDevice<ObjectType>(cu_params, args...);

		IsOk(cudaFree(cu_params));
		return cu_data;
	}

	template<typename ObjectType, typename ParamsType>
	__global__ static void KernelSyncParameters(ObjectType* cu_object, const ParamsType const* cu_params)
	{
		cu_object->OnSyncParameters(*cu_params);
	}

	template<typename ObjectType, typename ParamsType>
	__host__ static void SyncParameters(ObjectType* cu_object, const ParamsType& params)
	{
		Assert(cu_object);
		ParamsType* cu_params;
		IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
		IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

		KernelSyncParameters<<<1, 1>>>(cu_object, cu_params);

		IsOk(cudaFree(cu_params));
	}

	template<typename T>
	__global__ inline void KernelDestroyDeviceInstance(T* cu_instance)
	{
		if (cu_instance != nullptr) { delete cu_instance; }
	}

	template<typename T>
	__host__ inline void DestroyOnDevice(T** cu_data)
	{
		Assert(cu_data != nullptr);
		if (*cu_data == nullptr) { return; }
		
		KernelDestroyDeviceInstance << < 1, 1 >> > (*cu_data);		
		IsOk(cudaDeviceSynchronize());

		*cu_data = nullptr;
	}
}