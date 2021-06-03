#pragma once

#include "CudaVecBase.cuh"

namespace Cuda
{
	template<typename T>
	struct __builtin_align__(8) _ivec2 : public VecBase<2>
	{
		enum _attrs : size_t { kDims = 2 };
		using kType = T;

		union
		{
			struct { T x, y; };
			struct { T i0, i1; };
			T data[2];
		};

		_ivec2() = default;
		_ivec2(const _ivec2&) = default;
		__host__ __device__ explicit _ivec2(const T v) : x(v), y(v) {}
		__host__ __device__ _ivec2(const T& x_, const T& y_) : x(x_), y(y_) {}
		template<typename S, typename = std::enable_if<std::is_base_of<VecBase<2>, S>::value>::type>
		__host__ __device__ _ivec2(const S& other) : x(T(other.x)), y(T(other.y)) {}

		__host__ __device__ inline _ivec2& operator=(const T& v) { x = y = v; return *this; }

		__host__ __device__ inline const T& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline T& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%i, %i}", x, y); }
	};

	template<typename T> __host__ __device__ inline _ivec2<T> operator +(const _ivec2<T>& lhs, const _ivec2<T>& rhs) { return { lhs.x + rhs.x, lhs.y + rhs.y }; }
	template<typename T> __host__ __device__ inline _ivec2<T> operator -(const _ivec2<T>& lhs, const _ivec2<T>& rhs) { return { lhs.x - rhs.x, lhs.y - rhs.y }; }
	template<typename T> __host__ __device__ inline _ivec2<T> operator -(const _ivec2<T>& lhs) { return { -lhs.x, -lhs.y }; }
	template<typename T> __host__ __device__ inline _ivec2<T> operator *(const _ivec2<T>& lhs, const _ivec2<T>& rhs) { return { lhs.x * rhs.x, lhs.y * rhs.y }; }
	template<typename T> __host__ __device__ inline _ivec2<T> operator *(const _ivec2<T>& lhs, const int& rhs) { return { lhs.x * rhs, lhs.y * rhs }; }
	template<typename T> __host__ __device__ inline _ivec2<T> operator *(const int& lhs, const _ivec2<T>& rhs) { return { lhs * rhs.x, lhs * rhs.y }; }
	template<typename T> __host__ __device__ inline _ivec2<T> operator /(const _ivec2<T>& lhs, const int& rhs) { return { lhs.x / rhs, lhs.y / rhs }; }
	template<typename T> __host__ __device__ inline _ivec2<T>& operator +=(_ivec2<T>& lhs, const _ivec2<T>& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }
	template<typename T> __host__ __device__ inline _ivec2<T>& operator -=(_ivec2<T>& lhs, const _ivec2<T>& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; }
	template<typename T> __host__ __device__ inline _ivec2<T>& operator *=(_ivec2<T>& lhs, const int& rhs) { lhs.x *= rhs; lhs.y *= rhs; return lhs; }
	template<typename T> __host__ __device__ inline _ivec2<T>& operator /=(_ivec2<T>& lhs, const int& rhs) { lhs.x /= rhs; lhs.y /= rhs; return lhs; }
	template<typename T> __host__ __device__ inline _ivec2<T>& operator %=(_ivec2<T>& lhs, const int& rhs) { lhs.x %= rhs; lhs.y %= rhs; return lhs; }
	template<typename T> __host__ __device__ inline _ivec2<T>& operator %=(_ivec2<T>& lhs, const _ivec2<T>& rhs) { return { lhs.x % rhs.x, lhs.y % rhs.y }; }

	template<typename T, int ActualSize, int X, int Y>
	__host__ __device__ inline  __vec_swizzle<T, ActualSize, 2, X, Y>& operator +=(__vec_swizzle<T, ActualSize, 2, X, Y>& lhs, const _ivec2<T>& rhs)
	{
		lhs.data[X] += rhs.x; lhs.data[Y] += rhs.y; return lhs;
	}
	template<typename T, int ActualSize, int X, int Y>
	__host__ __device__ inline  __vec_swizzle<T, ActualSize, 2, X, Y>& operator -=(__vec_swizzle<T, ActualSize, 2, X, Y>& lhs, const _ivec2<T>& rhs)
	{
		lhs.data[X] -= rhs.x; lhs.data[Y] -= rhs.y; return lhs;
	}

	template<typename T> __host__ __device__ inline int dot(const _ivec2<T>& lhs, const _ivec2<T>& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }
	template<typename T> __host__ __device__ inline _ivec2<T> perpendicular(const _ivec2<T>& lhs) { return { -lhs.y, lhs.x }; }
	template<typename T, typename S = int> __host__ __device__ inline S length2(const _ivec2<T>& v) { return S(v.x * v.x + v.y * v.y); }
	template<typename T, typename S = float> __host__ __device__ inline S length(const _ivec2<T>& v) { return sqrt(S(v.x * v.x + v.y * v.y)); }
	template<typename T> __host__ __device__ inline _ivec2<T> clamp(const _ivec2<T>& v, const _ivec2<T>& a, const _ivec2<T>& b) { return { clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y) }; }
	template<typename T> __host__ __device__ inline _ivec2<T> abs(const _ivec2<T>& a) { return _ivec2<T>(abs(a.x), abs(a.y)); }
	template<typename T> __host__ __device__ inline T sum(const _ivec2<T>& a) { return a.x + a.y; }

	template<typename T> __host__ __device__ inline bool operator==(const _ivec2<T>& lhs, const _ivec2<T>& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }
	template<typename T> __host__ __device__ inline bool operator!=(const _ivec2<T>& lhs, const _ivec2<T>& rhs) { return lhs.x != rhs.x || lhs.y != rhs.y; }

	using ivec2 = _ivec2<int>;
	using uvec2 = _ivec2<unsigned int>;
}