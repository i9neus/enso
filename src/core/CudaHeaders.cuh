#pragma once

#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include "thirdparty/nvidia/helper_cuda.h"

#include "math/Constants.h"
#include "Assert.h"
#include <type_traits>
#include <math.h>

#define kKernelX				(blockIdx.x * blockDim.x + threadIdx.x)	
#define kKernelY				(blockIdx.y * blockDim.y + threadIdx.y)	
#define kKernelIdx				kKernelX
#define kThreadIdx				threadIdx.x
#define kBlockIdx				blockIdx.x
#define kWarpLane				(threadIdx.x & 31)
#define kKernelWidth			(gridDim.x * blockDim.x)
#define kKernelHeight			(gridDim.y * blockDim.y)
#define kIsFirstThread			(threadIdx.x == 0 && threadIdx.y == 0)

template<typename T> __device__ __forceinline__ T kKernelPos() { return T(typename T::kType(kKernelX), typename T::kType(kKernelY)); }
template<typename T> __device__ __forceinline__ T kKernelDims() { return T(typename T::kType(kKernelWidth), typename T::kType(kKernelHeight)); }

//#define CUDA_DEVICE_GLOBAL_ASSERTS

#ifdef _DEBUG
#define dassert(condition) assert(condition)
#else
#define dassert(condition)
#endif

namespace Enso
{	    
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
#ifdef _DEBUG
        size_t oldLimit;
        IsOk(cudaDeviceGetLimit(&oldLimit, cudaLimitStackSize));
        IsOk(cudaDeviceSetLimit(cudaLimitStackSize, newLimit));
        IsOk(cudaDeviceSynchronize());

        functor();

        IsOk(cudaDeviceSetLimit(cudaLimitStackSize, oldLimit));
        IsOk(cudaDeviceSynchronize());
#else
        functor();
#endif
    }
}