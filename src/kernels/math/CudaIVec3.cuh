#pragma once

#include "../CudaCommonIncludes.cuh"
#include "generic/Constants.h"

namespace Cuda
{
	template<typename T>
	struct _ivec3 : public VecBase<3>
	{
		enum _attrs : size_t { kDims = 3 };
		using kType = T;

		union
		{
			struct { T x, y, z; };
			struct { T i0, i1, i2; };
			T data[3];
		};

		__host__ __device__ _ivec3() = default;
		__host__ __device__ _ivec3(const T v) : x(v), y(v), z(v) {}
		__host__ __device__ _ivec3(const T & x_, const T & y_, const T& z_) : x(x_), y(y_), z(z_) {}
		__host__ __device__ _ivec3(const _ivec2<T>& v, const float& z_) : x(v.x), y(v.y), z(z_) {}
		template<typename S, typename = std::enable_if<std::is_base_of<VecBase<3>, S>::value>::type>
		__host__ __device__ _ivec3(const S& other) : x(T(other.x)), y(T(other.y)), z(T(other.z)) {}

		__host__ __device__ inline const T& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline T& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%i, %i, %i}", x, y, z); }
	};

	template<typename T> __host__ __device__ inline _ivec3<T> operator +(const _ivec3<T>& lhs, const _ivec3<T>& rhs) { return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z }; }
	template<typename T> __host__ __device__ inline _ivec3<T> operator -(const _ivec3<T>& lhs, const _ivec3<T>& rhs) { return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z }; }
	template<typename T> __host__ __device__ inline _ivec3<T> operator -(const _ivec3<T>& lhs) { return { -lhs.x, -lhs.y, -lhs-z }; }
	template<typename T> __host__ __device__ inline _ivec3<T> operator *(const _ivec3<T>& lhs, const _ivec3<T>& rhs) { return { lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z }; }
	template<typename T> __host__ __device__ inline _ivec3<T> operator *(const _ivec3<T>& lhs, const int& rhs) { return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs }; }
	template<typename T> __host__ __device__ inline _ivec3<T> operator *(const int& lhs, const _ivec3<T>& rhs) { return { lhs * rhs.x, lhs * rhs.y, lhs * rhs.z }; }
	template<typename T> __host__ __device__ inline _ivec3<T> operator /(const _ivec3<T>& lhs, const int& rhs) { return { lhs.x / rhs, lhs.y / rhs, lhs.z / rhs }; }
	template<typename T> __host__ __device__ inline _ivec3<T>& operator +=(_ivec3<T>& lhs, const _ivec3<T>& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }
	template<typename T> __host__ __device__ inline _ivec3<T>& operator -=(_ivec3<T>& lhs, const _ivec3<T>& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }
	template<typename T> __host__ __device__ inline _ivec3<T>& operator *=(_ivec3<T>& lhs, const int& rhs) { lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; return lhs; }
	template<typename T> __host__ __device__ inline _ivec3<T>& operator /=(_ivec3<T>& lhs, const int& rhs) { lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs; return lhs; }
	template<typename T> __host__ __device__ inline _ivec3<T>& operator %=(_ivec3<T>& lhs, const int& rhs) { lhs.x %= rhs; lhs.y %= rhs; lhs.z %= rhs; return lhs; }
	template<typename T> __host__ __device__ inline _ivec3<T>& operator %=(_ivec3<T>& lhs, const _ivec3<T>& rhs) { return { lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z }; }

	template<typename T> __host__ __device__ inline int dot(const _ivec3<T>& lhs, const _ivec3<T>& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
	template<typename T, typename S = int> __host__ __device__ inline S length2(const _ivec3<T>& v) { return S(v.x * v.x + v.y * v.y + v.z * v.z); }
	template<typename T, typename S = float> __host__ __device__ inline S length(const _ivec3<T>& v) { return sqrt(S(v.x * v.x + v.y * v.y + v.z * v.z)); }
	template<typename T> __host__ __device__ inline _ivec3<T> clamp(const _ivec3<T>& v, const _ivec3<T>& a, const _ivec3<T>& b) { return { clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z) }; }

	using ivec3 = _ivec3<int>;
	using uvec3 = _ivec3<unsigned int>;
}