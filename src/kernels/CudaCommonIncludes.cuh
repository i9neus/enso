#pragma once

#include <cuda_runtime.h>
#include "cuda_fp16.h"

#include "Assert.h"
#include "generic/Asset.h"
#include "generic/StdIncludes.h"
#include "generic/thirdparty/nvidia/helper_cuda.h"

namespace Cuda
{
	using uint = unsigned int;
	
	__host__ __device__ inline float clamp(const float& v, const float& a, const float& b) { return fmaxf(a, fminf(v, b)); }
	__host__ __device__ inline float fract(const float& v) { return fmodf(v, 1.0f); }
	__host__ __device__ inline float sign(const float& v) { return copysign(1.0f, v); }
	template<typename T> __host__ inline void echo(const T& t) { std::printf("%s\n", t.format().c_str()); }

	template<int T> struct VecBase {};

	#define KERNEL_COORDS_IVEC2 ivec2(blockIdx.x* blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)

	template<typename T>
	inline void SafeFreeDeviceMemory(T** deviceData)
	{
		Assert(deviceData);
		if (*deviceData != nullptr)
		{
			checkCudaErrors(cudaFree(*deviceData));
			*deviceData = nullptr;
		}
	}

	template<typename T>
	inline void SafeCreateDeviceInstance(T** deviceData, const T* instance)
	{
		Assert(deviceData);
		Assert(*deviceData == nullptr);
		
		checkCudaErrors(cudaMalloc((void**)deviceData, sizeof(T)));
		checkCudaErrors(cudaMemcpy(*deviceData, instance, sizeof(T), cudaMemcpyHostToDevice));
	}
}