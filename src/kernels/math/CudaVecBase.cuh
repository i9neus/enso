#pragma once

#include "../CudaCommonIncludes.cuh"
#include "generic/Constants.h"

namespace Cuda
{
	template<typename Type, int ActualSize, int SpoofedSize, int... Indices>
	class __vec_swizzle
	{
	public:
		Type data[ActualSize];

	private:
		template<int J, int K, int... Kn>
		__host__ __device__ inline
			typename std::enable_if<sizeof...(Kn) == 0>::type unpackRecurse(Type* vecData) const
		{
			vecData[J] = data[K];
		}

		template<int J, int K, int... Kn>
		__host__ __device__ inline
			typename std::enable_if<sizeof...(Kn) != 0>::type unpackRecurse(Type* vecData) const
		{
			vecData[J] = data[K];
			unpackRecurse<J + 1, Kn...>(vecData);
		}

	public:
		__vec_swizzle() = default;

		__host__ __device__ inline void unpack(Type* vecData) const
		{
			unpackRecurse<0, Indices...>(vecData);
		}
	};

	template<typename Type, int ActualSize, int... Indices>
	__host__ __device__ inline  __vec_swizzle<Type, ActualSize, 2, Indices...>& operator *=(__vec_swizzle<Type, ActualSize, 2, Indices...>& lhs, const Type& rhs)
	{
		lhs.data[0] *= rhs;	lhs.data[1] *= rhs;	return lhs;
	}
	template<typename Type, int ActualSize, int... Indices>
	__host__ __device__ inline  __vec_swizzle<Type, ActualSize, 3, Indices...>& operator *=(__vec_swizzle<Type, ActualSize, 3, Indices...>& lhs, const Type& rhs)
	{
		lhs.data[0] *= rhs;	lhs.data[1] *= rhs;	lhs.data[2] *= rhs;	return lhs;
	}
	template<typename Type, int ActualSize, int... Indices>
	__host__ __device__ inline  __vec_swizzle<Type, ActualSize, 4, Indices...>& operator *=(__vec_swizzle<Type, ActualSize, 4, Indices...>& lhs, const Type& rhs)
	{
		lhs.data[0] *= rhs;	lhs.data[1] *= rhs;	lhs.data[2] *= rhs;	lhs.data[3] *= rhs;	return lhs;
	}

	template<typename Type, int ActualSize, int... Indices>
	__host__ __device__ inline  __vec_swizzle<Type, ActualSize, 2, Indices...>& operator /=(__vec_swizzle<Type, ActualSize, 2, Indices...>& lhs, const Type& rhs)
	{
		lhs.data[0] /= rhs;	lhs.data[1] /= rhs;	return lhs;
	}
	template<typename Type, int ActualSize, int... Indices>
	__host__ __device__ inline  __vec_swizzle<Type, ActualSize, 3, Indices...>& operator /=(__vec_swizzle<Type, ActualSize, 3, Indices...>& lhs, const Type& rhs)
	{
		lhs.data[0] /= rhs;	lhs.data[1] /= rhs;	lhs.data[2] /= rhs;	return lhs;
	}
	template<typename Type, int ActualSize, int... Indices>
	__host__ __device__ inline  __vec_swizzle<Type, ActualSize, 4, Indices...>& operator /=(__vec_swizzle<Type, ActualSize, 4, Indices...>& lhs, const Type& rhs)
	{
		lhs.data[0] /= rhs;	lhs.data[1] /= rhs;	lhs.data[2] /= rhs;	lhs.data[3] /= rhs;	return lhs;
	}

	template<int T> struct VecBase {};	
	

}