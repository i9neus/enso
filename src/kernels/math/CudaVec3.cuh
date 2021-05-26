#pragma once

#include "../CudaCommonIncludes.cuh"
#include "generic/Constants.h"
#include "CudaVec2.cuh""

namespace Cuda
{
	struct vec3 : public VecBase<3>
	{
		enum _attrs : size_t { kDims = 3 };
		using kType = float;

		union
		{
			struct { float x, y, z; };
			struct { float i0, i1, i2; };
			float data[3];
		};

		__host__ __device__ vec3() = default;
		__host__ __device__ vec3(const float v) : x(v), y(v), z(v) {}
		__host__ __device__ vec3(const float& x_, const float& y_, const float& z_) : x(x_), y(y_), z(z_) {}
		__host__ __device__ vec3(const vec2& v, const float& z_) : x(v.x), y(v.y), z(z_) {}
		__host__ __device__ vec3(const vec3& other) : x(other.x), y(other.y), z(other.z) {}
		template<typename T, typename = std::enable_if<std::is_base_of<VecBase<3>, T>::value>::type>
		__host__ __device__ vec3(const T& other) : x(float(other.x)), y(float(other.y)), z(float(other.z)) {}

		__host__ __device__ inline const float& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline float& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%f, %f, %f}", x, y, z); }
	};

	__host__ __device__ inline vec3 operator +(const vec3& lhs, const vec3& rhs) { return vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
	__host__ __device__ inline vec3 operator -(const vec3& lhs, const vec3& rhs) { return vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
	__host__ __device__ inline vec3 operator -(const vec3& lhs) { return vec3(-lhs.x, -lhs.y, -lhs.z); }
	__host__ __device__ inline vec3 operator *(const vec3& lhs, const vec3& rhs) { return vec3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
	__host__ __device__ inline vec3 operator *(const vec3& lhs, const float& rhs) { return vec3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs); }
	__host__ __device__ inline vec3 operator *(const float& lhs, const vec3& rhs) { return vec3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z); }
	__host__ __device__ inline vec3 operator /(const vec3& lhs, const float& rhs) { return vec3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs); }
	__host__ __device__ inline vec3& operator +=(vec3& lhs, const vec3& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }
	__host__ __device__ inline vec3& operator -=(vec3& lhs, const vec3& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }
	__host__ __device__ inline vec3& operator *=(vec3& lhs, const float& rhs) { lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; return lhs; }
	__host__ __device__ inline vec3& operator /=(vec3& lhs, const float& rhs) { lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs; return lhs; }

	__host__ __device__ inline float dot(const vec3& lhs, const vec3& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
	__host__ __device__ inline vec3 cross(const vec3& lhs, const vec3& rhs)
	{
		return vec3(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x);
	}
	__host__ __device__ inline float length2(const vec3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
	__host__ __device__ inline float length(const vec3& v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
	__host__ __device__ inline vec3 normalize(const vec3& v) { return v / length(v); }
	__host__ __device__ inline vec3 fmod(const vec3& a, const vec3& b) { return vec3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z)); }
	__host__ __device__ inline vec3 pow(const vec3& a, const vec3& b) { return vec3(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z)); }
	__host__ __device__ inline vec3 exp(const vec3& a) { return vec3(expf(a.x), expf(a.y), expf(a.z)); }
	__host__ __device__ inline vec3 log(const vec3& a) { return vec3(logf(a.x), logf(a.y), logf(a.z)); }
	__host__ __device__ inline vec3 log10(const vec3& a) { return vec3(log10f(a.x), log10f(a.y), log10f(a.z)); }
	__host__ __device__ inline vec3 log2(const vec3& a) { return vec3(log2f(a.x), log2f(a.y), log2f(a.z)); }
	__host__ __device__ inline vec3 clamp(const vec3& v, const vec3& a, const vec3& b) { return vec3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }
	__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b) { return vec3(clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f), clamp(v.z, 0.0f, 1.0f)); }
	// FIXME: Cuda intrinsics aren't working. Why is this?
	//__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }
}