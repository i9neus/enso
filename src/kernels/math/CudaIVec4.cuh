#pragma once

#include "../CudaCommonIncludes.cuh"
#include "generic/Constants.h"

namespace Cuda
{
	template<typename T>
	struct __builtin_align__(8) _ivec4
	{
		enum _attrs : size_t { kDims = 4 };

		union
		{
			struct { T x, y, z, w; };
			struct { T i0, i1, i2, i3; };
			T data[4];
		};

		_ivec4() = default;
		_ivec4(const T v) : x(v), y(v), z(v), w(v) {}
		_ivec4(const T & x_, const T & y_, const T & z_, const T& w_) : x(x_), y(y_), z(z_), w(w_) {}

		__host__ __device__ inline const T& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline T& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%i, %i, %i, %i}", x, y, z, w); }
	};

	template<typename T> __host__ __device__ inline _ivec4<T> operator +(const _ivec4<T>& lhs, const _ivec4<T>& rhs) { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w }; }
	template<typename T> __host__ __device__ inline _ivec4<T> operator -(const _ivec4<T>& lhs, const _ivec4<T>& rhs) { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w }; }
	template<typename T> __host__ __device__ inline _ivec4<T> operator -(const _ivec4<T>& lhs) { return { -lhs.x, -lhs.y, -lhs.z, -lhs.w }; }
	template<typename T> __host__ __device__ inline _ivec4<T> operator *(const _ivec4<T>& lhs, const _ivec4<T>& rhs) { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w }; }
	template<typename T> __host__ __device__ inline _ivec4<T> operator *(const _ivec4<T>& lhs, const int& rhs) { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs }; }
	template<typename T> __host__ __device__ inline _ivec4<T> operator *(const int& lhs, const _ivec4<T>& rhs) { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w }; }
	template<typename T> __host__ __device__ inline _ivec4<T> operator /(const _ivec4<T>& lhs, const int& rhs) { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs }; }
	template<typename T> __host__ __device__ inline _ivec4<T>& operator +=(_ivec4<T>& lhs, const _ivec4<T>& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs; }
	template<typename T> __host__ __device__ inline _ivec4<T>& operator -=(_ivec4<T>& lhs, const _ivec4<T>& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs; }
	template<typename T> __host__ __device__ inline _ivec4<T>& operator *=(_ivec4<T>& lhs, const int& rhs) { lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; lhs.w *= rhs; return lhs; }
	template<typename T> __host__ __device__ inline _ivec4<T>& operator /=(_ivec4<T>& lhs, const int& rhs) { lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs; lhs.w /= rhs; return lhs; }
	template<typename T> __host__ __device__ inline _ivec4<T>& operator %=(_ivec4<T>& lhs, const int& rhs) { lhs.x %= rhs; lhs.y %= rhs; lhs.z %= rhs; lhs.w %= rhs; return lhs; }
	template<typename T> __host__ __device__ inline _ivec4<T>& operator %=(_ivec4<T>& lhs, const _ivec4<T>& rhs) { return { lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z, lhs.w %= rhs.w }; }

	template<typename T> __host__ __device__ inline int dot(const _ivec4<T>& lhs, const _ivec4<T>& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w; }
	template<typename T, typename S = int> __host__ __device__ inline S length2(const _ivec4<T>& v) { return S(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w); }
	template<typename T, typename S = float> __host__ __device__ inline S length(const _ivec4<T>& v) { return sqrt(S(v.x * v.x + v.y * v.y + v.z * v.z + v.z * v.w)); }
	template<typename T> __host__ __device__ inline _ivec4<T> clamp(const _ivec4<T>& v, const _ivec4<T>& a, const _ivec4<T>& b) { return { clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w) }; }

	using ivec4 = _ivec4<int>;
	using uvec4 = _ivec4<unsigned int>;
}