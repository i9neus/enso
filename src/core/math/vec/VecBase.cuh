#pragma once

#include "../../CudaHeaders.cuh"
#include "../MathUtils.cuh"

namespace Enso
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
		__host__ __device__ __vec_swizzle() {}

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
}