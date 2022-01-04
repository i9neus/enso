#pragma once

#include "../../CudaCommonIncludes.cuh"
#include "generic/Constants.h"

namespace Cuda
{	
	template<typename Type, int ActualSize, int SpoofedSize, int... Indices>
	struct __vec_swizzle
	{
		template<typename OtherType, int OtherActualSize, int OtherSpoofedSize, int... OtherIndices> friend struct __vec_swizzle;

		enum _attrs : size_t { kDims = SpoofedSize };
		Type data[ActualSize];

	private:
		template<int L0, int L1, int R0, int R1>
		__host__ __device__ __forceinline__ void UnpackTo(Type* otherData) const
		{
			otherData[L0] = data[R0];
			otherData[L1] = data[R1];
		}

		template<int L0, int L1, int L2, int R0, int R1, int R2>
		__host__ __device__ __forceinline__ void UnpackTo(Type* otherData) const
		{
			otherData[L0] = data[R0];
			UnpackTo<L1, L2, R1, R2>(otherData);
		}

		template<int L0, int L1, int L2, int L3, int R0, int R1, int R2, int R3>
		__host__ __device__ __forceinline__ void UnpackTo(Type* otherData) const
		{
			otherData[L0] = data[R0];
			UnpackTo<L1, L2, L3, R1, R2, R3>(otherData);
		}

	public:
		__vec_swizzle() = default;
		__vec_swizzle(const __vec_swizzle& other) = default;

		// Assign from swizzled types
		template<int OtherActualSize, int... OtherIndices>
		__host__ __device__ __forceinline__ __vec_swizzle& operator=(const __vec_swizzle<Type, OtherActualSize, SpoofedSize, OtherIndices...>& rhs)
		{
			rhs.UnpackTo<Indices..., OtherIndices...>(data);
			return *this;
		}

		// Assign from arithmetic types
		template<typename OtherType, typename = typename std::enable_if<std::is_arithmetic<OtherType>::value>::type>
		__host__ __device__ __forceinline__ __vec_swizzle& operator=(const OtherType& rhs)
		{
			for (int i = 0; i < ActualSize; i++)
			{
				data[i] = Type(rhs);
			}
			return *this;
		}
	};

	// Generic vector types for more convenient templatisation
	template<typename T> using Tvec4 = __vec_swizzle<T, 4, 4, 0, 1, 2, 3>;
	template<typename T> using Tvec3 = __vec_swizzle<T, 3, 3, 0, 1, 2>;
	template<typename T> using Tvec2 = __vec_swizzle<T, 2, 2, 0, 1>;

	template<typename T> __host__ __device__ __forceinline__ T max(const T& a, const T& b) { return (a > b) ? a : b; }
	template<typename T> __host__ __device__ __forceinline__ T min(const T& a, const T& b) { return (a < b) ? a : b; }
	__host__ __device__ __forceinline__ float clamp(const float& v, const float& a, const float& b) noexcept { return fmaxf(a, fminf(v, b)); }
	template<typename T> __host__ __device__ __forceinline__ T clamp(const T& v, const T& a, const T& b) noexcept { return max(a, min(v, b)); }
	__host__ __device__ __forceinline__ float fract(const float& v) noexcept { return fmodf(v, 1.0f); }
	__host__ __device__ __forceinline__ float sign(const float& v) noexcept { return copysign(1.0f, v); }

	namespace math
	{
#define USE_CUDA_INTRINSICS

#if defined(__CUDA_ARCH__) && defined(USE_CUDA_INTRINSICS)
		template<typename T> __device__ __forceinline__ T cos(const T& v) { return __cosf(v); }
		template<typename T> __device__ __forceinline__ T sin(const T& v) { return __sinf(v); }
		template<typename T> __device__ __forceinline__ T tan(const T& v) { return __tanf(v); }
		template<typename T> __device__ __forceinline__ T exp(const T& v) { return __expf(v); }
		template<typename T> __device__ __forceinline__ T sqrt(const T& v) { return __fsqrt_rn(v); }
		template<typename T> __device__ __forceinline__ T pow(const T& a, const T& b) { return __powf(a, b); }
		template<typename T> __device__ __forceinline__ T log10(const T& v) { return __log10f(v); }
		template<typename T> __device__ __forceinline__ T log2(const T& v) { return __log2f(v); }
		template<typename T> __device__ __forceinline__ T log(const T& v) { return __logf(v); }
		template<typename T> __device__ __forceinline__ void sincos(const T& v, T& s, T& c) { __sincosf(v, &s, &c); }
#else
		template<typename T> __host__ __device__ __forceinline__ T cos(const T& v) noexcept { return cosf(v); }
		template<typename T> __host__ __device__ __forceinline__ T sin(const T& v) noexcept { return sinf(v); }
		template<typename T> __host__ __device__ __forceinline__ T tan(const T& v) noexcept { return tanf(v); }
		template<typename T> __host__ __device__ __forceinline__ T exp(const T& v) noexcept { return expf(v); }
		template<typename T> __host__ __device__ __forceinline__ T sqrt(const T& v) { return sqrtf(v); }
		template<typename T> __host__ __device__ __forceinline__ T pow(const T& a, const T& b) noexcept { return powf(a, b); }
		template<typename T> __host__ __device__ __forceinline__ T log10(const T& v) noexcept { return log10f(v); }
		template<typename T> __host__ __device__ __forceinline__ T log2(const T& v) noexcept { return log2f(v); }
		template<typename T> __host__ __device__ __forceinline__ T log(const T& v) noexcept { return logf(v); }
		template<typename T> __host__ __device__ __forceinline__ void sincos(const T& v, T& s, T& c) noexcept
		{ 
			s = sinf(v);
			c = cosf(v);
		}
#endif
	}

}