#pragma once

#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include "thirdparty/nvidia/helper_cuda.h"

#include "generic/Constants.h"
#include "generic/Assert.h"
#include <type_traits>

// saturate() causes a compiler error when called from a function with both __host__ and __device decorators. Use saturatef() instead.
#ifdef __CUDA_ARCH__
#define saturatef(a) saturate(a)
#else
inline float saturatef(const float a) { return (a < 0.0f) ? 0.0f : ((a > 1.0f) ? 1.0f : a); }
#endif

// Define any CUDA math functions that aren't defined in libraries like cmath
#ifndef __CUDACC__
#include <math.h>

// Define implementations of min and max for integral types in the root namespace
template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type> 
__host__ __device__ inline T max(const T a, const T b) { return a > b ? a : b; }
template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
__host__ __device__ inline T min(const T a, const T b) { return a < b ? a : b; }

#endif

//#define CUDA_DEVICE_GLOBAL_ASSERTS

#ifdef _DEBUG
#define dassert(condition) assert(condition)
#else
#define dassert(condition)
#endif

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

    // Little RAII class that temporarily increases the stack size and returns it to the default
    // when the object is destroyed.
    class DeviceStackManager
    {
    public:
        DeviceStackManager(const size_t newLimit)
        {
            IsOk(cudaDeviceGetLimit(&m_oldLimit, cudaLimitStackSize));
            IsOk(cudaDeviceSetLimit(cudaLimitStackSize, newLimit));
            IsOk(cudaDeviceSynchronize());
        }

        ~DeviceStackManager()
        {
            cudaDeviceSetLimit(cudaLimitStackSize, m_oldLimit);
            cudaDeviceSynchronize();
        }

    private:
        size_t m_oldLimit;
    };

    // RAII functor that temporarily increases the stack size, calls the functor, then returns the stack size to its original value
    template<typename Lambda>
    __host__ inline void ScopedDeviceStackResize(const size_t newLimit, Lambda functor)
    {
        size_t oldLimit;
        IsOk(cudaDeviceGetLimit(&oldLimit, cudaLimitStackSize));
        IsOk(cudaDeviceSetLimit(cudaLimitStackSize, newLimit));
        IsOk(cudaDeviceSynchronize());

        functor();

        IsOk(cudaDeviceSetLimit(cudaLimitStackSize, oldLimit));
        IsOk(cudaDeviceSynchronize());
    }
}