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