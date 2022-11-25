#include "CudaTests.cuh"
#include "core/math/Math.cuh"
#include "io/Log.h"

namespace Enso
{
	struct TypeSizeTestResults
	{
		uint vec2Size;
		uint vec3Size;
		uint vec4Size;
		uint ivec2Size;
		uint ivec3Size;
		uint ivec4Size;
		uint mat4Size;
		uint mat3Size;
		uint ptrSize;

		uint kifsScratchpadSize;
	};

	__global__ void KernelVerifyTypeSizes(TypeSizeTestResults* results)
	{
		results->vec2Size = sizeof(vec2);
		results->vec3Size = sizeof(vec3);
		results->vec4Size = sizeof(vec4);
		results->ivec2Size = sizeof(ivec2);
		results->ivec3Size = sizeof(ivec3);
		results->ivec4Size = sizeof(ivec4);
		results->mat4Size = sizeof(mat4);
		results->mat3Size = sizeof(mat3);
		results->ptrSize = sizeof(int*);
	}

	__host__ void VerifyTypeSizes()
	{
		TypeSizeTestResults* cu_resultsBuffer;
		IsOk(cudaMalloc((void***)&cu_resultsBuffer, sizeof(TypeSizeTestResults)));

		KernelVerifyTypeSizes << < 1, 1 >> > (cu_resultsBuffer);
		IsOk(cudaDeviceSynchronize());

		TypeSizeTestResults results;
		IsOk(cudaMemcpy(&results, cu_resultsBuffer, sizeof(TypeSizeTestResults), cudaMemcpyDeviceToHost));
		IsOk(cudaFree(cu_resultsBuffer));

		AssertMsgFmt(results.vec2Size == sizeof(vec2), "vec2 host/device size mismatch. Host: %i bytes, device %i bytes", sizeof(vec2), results.vec2Size);
		AssertMsgFmt(results.vec3Size == sizeof(vec3), "vec3 host/device size mismatch. Host: %i bytes, device %i bytes", sizeof(vec3), results.vec3Size);
		AssertMsgFmt(results.vec4Size == sizeof(vec4), "vec4 host/device size mismatch. Host: %i bytes, device %i bytes", sizeof(vec4), results.vec4Size);
		AssertMsgFmt(results.ivec2Size == sizeof(ivec2), "ivec2 host/device size mismatch. Host: %i bytes, device %i bytes", sizeof(ivec2), results.ivec2Size);
		AssertMsgFmt(results.ivec3Size == sizeof(ivec3), "ivec3 host/device size mismatch. Host: %i bytes, device %i bytes", sizeof(ivec3), results.ivec3Size);
		AssertMsgFmt(results.ivec4Size == sizeof(ivec4), "ivec4 host/device size mismatch. Host: %i bytes, device %i bytes", sizeof(ivec4), results.ivec4Size);
		AssertMsgFmt(results.mat3Size == sizeof(mat3), "mat3 host/device size mismatch. Host: %i bytes, device %i bytes", sizeof(mat3), results.mat3Size);
		AssertMsgFmt(results.mat4Size == sizeof(mat4), "mat4 host/device size mismatch. Host: %i bytes, device %i bytes", sizeof(mat4), results.mat4Size);

		Log::Debug("Type size information:\n");
		{
			Log::Indent indent;
			Log::Debug("- Device::KIFS::Scratchpad: %i bytes\n", results.kifsScratchpadSize);
			Log::Debug("- Device pointer: %i bytes\n", results.ptrSize);
		}

		Log::Write("Type size check passed!\n");
	} 

	__global__ void KernelTestScheduling(uint* results)
	{
		/*if (kKernelIdx > 16)
		{
			return;
		}*/
		
		__syncthreads();

		*results = __activemask();
	}
}