#pragma once

#include "../../CudaCommonIncludes.cuh"
#include "generic/Constants.h"

namespace Cuda
{	
	template<typename Type, int ActualSize, int SpoofedSize, int... Indices>
	struct __vec_swizzle
	{
		template<typename OtherType, int OtherActualSize, int OtherSpoofedSize, int... OtherIndices> friend struct __vec_swizzle;

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

	__host__ __device__ __forceinline__ float clamp(const float& v, const float& a, const float& b) { return fmaxf(a, fminf(v, b)); }
	__host__ __device__ __forceinline__ float fract(const float& v) { return fmodf(v, 1.0f); }
	__host__ __device__ __forceinline__ float sign(const float& v) { return copysign(1.0f, v); }
}