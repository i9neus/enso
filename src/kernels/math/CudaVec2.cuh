#pragma once

#include "../CudaCommonIncludes.cuh"
#include "generic/Constants.h"

namespace Cuda
{
	struct __builtin_align__(8) vec2 : public VecBase<2>
	{
		enum _attrs : size_t { kDims = 2 };
		using kType = float;

		union
		{
			struct { float x, y; };
			struct { float i0, i1; };
			float data[2];
		};

		vec2() = default;
		__host__ __device__ vec2(const float v) : x(v), y(v) {}
		__host__ __device__ vec2(const float& x_, const float& y_) : x(x_), y(y_) {}
		__host__ __device__ vec2(const vec2 & other) : x(other.x), y(other.y) {}
		template<typename T, typename = std::enable_if<std::is_base_of<VecBase<2>, T>::value>::type>
		__host__ __device__ vec2(const T& other) : x(float(other.x)), y(float(other.y)) {}

		__host__ __device__ inline const float& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline float& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%f, %f}", x, y); }
	};

	__host__ __device__ inline vec2 operator +(const vec2& lhs, const vec2& rhs) { return vec2(lhs.x + rhs.x, lhs.y + rhs.y); }
	__host__ __device__ inline vec2 operator -(const vec2& lhs, const vec2& rhs) { return vec2(lhs.x - rhs.x, lhs.y - rhs.y); }
	__host__ __device__ inline vec2 operator -(const vec2& lhs) { return vec2(-lhs.x, -lhs.y); }
	__host__ __device__ inline vec2 operator *(const vec2& lhs, const vec2& rhs) { return vec2(lhs.x * rhs.x, lhs.y * rhs.y); }
	__host__ __device__ inline vec2 operator *(const vec2& lhs, const float& rhs) { return vec2(lhs.x * rhs, lhs.y * rhs); }
	__host__ __device__ inline vec2 operator *(const float& lhs, const vec2& rhs) { return vec2(lhs * rhs.x, lhs * rhs.y); }
	__host__ __device__ inline vec2 operator /(const vec2& lhs, const float& rhs) { return vec2(lhs.x / rhs, lhs.y / rhs); }
	__host__ __device__ inline vec2& operator +=(vec2& lhs, const vec2& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; ; return lhs; }
	__host__ __device__ inline vec2& operator -=(vec2& lhs, const vec2& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y;  return lhs; }
	__host__ __device__ inline vec2& operator *=(vec2& lhs, const float& rhs) { lhs.x *= rhs; lhs.y *= rhs; return lhs; }
	__host__ __device__ inline vec2& operator /=(vec2& lhs, const float& rhs) { lhs.x /= rhs; lhs.y /= rhs; return lhs; }

	__host__ __device__ inline float dot(const vec2& lhs, const vec2& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }
	__host__ __device__ inline vec2 perpendicular(const vec2& lhs) { return vec2(-lhs.y, lhs.x); }
	__host__ __device__ inline float length2(const vec2& v) { return v.x * v.x + v.y * v.y; }
	__host__ __device__ inline float length(const vec2& v) { return sqrt(v.x * v.x + v.y * v.y); }
	__host__ __device__ inline vec2 normalize(const vec2& v) { return v / length(v); }
	__host__ __device__ inline vec2 fmod(const vec2& a, const vec2& b) { return vec2(fmodf(a.x, b.x), fmodf(a.y, b.y)); }
	__host__ __device__ inline vec2 pow(const vec2& a, const vec2& b) { return vec2(powf(a.x, b.x), powf(a.y, b.y)); }
	__host__ __device__ inline vec2 exp(const vec2& a) { return vec2(expf(a.x), expf(a.y)); }
	__host__ __device__ inline vec2 log(const vec2& a) { return vec2(logf(a.x), logf(a.y)); }
	__host__ __device__ inline vec2 log10(const vec2& a) { return vec2(log10f(a.x), log10f(a.y)); }
	__host__ __device__ inline vec2 log2(const vec2& a) { return vec2(log2f(a.x), log2f(a.y)); }
	__host__ __device__ inline vec2 clamp(const vec2& v, const vec2& a, const vec2& b) { return vec2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
	__host__ __device__ inline vec2 saturate(const vec2& v, const vec2& a, const vec2& b) { return vec2(clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f)); }
	// FIXME: Cuda intrinsics aren't working. Why is this?
	//__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }
}