#pragma once

#include "CudaVecBase.cuh"

namespace Cuda
{
	struct __align__(8) vec2 : public VecBase<2>
	{
		enum _attrs : size_t { kDims = 2 };
		using kType = float;
		union  
		{
			struct { float x, y; };
			struct { float i0, i1; };
			float data[2];

			__vec_swizzle<float, 2, 2, 0, 0> xx;
			__vec_swizzle<float, 2, 2, 0, 1> xy;
			__vec_swizzle<float, 2, 2, 1, 0> yx;
			__vec_swizzle<float, 2, 2, 1, 1> yy;
		};

		vec2() = default;
		vec2(const vec2&) = default;
		__host__ __device__ explicit vec2(const float v) : x(v), y(v) {}
		__host__ __device__ vec2(const float& x_, const float& y_) : x(x_), y(y_) {}
		template<typename T, typename = std::enable_if<std::is_base_of<VecBase<2>, T>::value>::type>
		__host__ __device__ vec2(const T& other) : x(float(other.x)), y(float(other.y)) {}

		template<int ActualSize, int... In>
		__host__ __device__ inline vec2(const __vec_swizzle<float, ActualSize, 2, In...>& swizzled) { swizzled.unpack(data); }

		__host__ __device__ vec2& operator=(const float& v) { x = v; y = v; return *this; }

		__host__ __device__ inline const float& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline float& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%.10f, %.10f}", x, y); }
	};

	__host__ __device__ inline vec2 operator +(const vec2& lhs, const vec2& rhs) { return vec2(lhs.x + rhs.x, lhs.y + rhs.y); }
	__host__ __device__ inline vec2 operator +(const vec2& lhs, const float& rhs) { return vec2(lhs.x + rhs, lhs.y + rhs); }
	__host__ __device__ inline vec2 operator -(const vec2& lhs, const vec2& rhs) { return vec2(lhs.x - rhs.x, lhs.y - rhs.y); }
	__host__ __device__ inline vec2 operator -(const vec2& lhs, const float& rhs) { return vec2(lhs.x - rhs, lhs.y - rhs); }
	__host__ __device__ inline vec2 operator -(const vec2& lhs) { return vec2(-lhs.x, -lhs.y); }
	__host__ __device__ inline vec2 operator *(const vec2& lhs, const vec2& rhs) { return vec2(lhs.x * rhs.x, lhs.y * rhs.y); }
	__host__ __device__ inline vec2 operator *(const vec2& lhs, const float& rhs) { return vec2(lhs.x * rhs, lhs.y * rhs); }
	__host__ __device__ inline vec2 operator *(const float& lhs, const vec2& rhs) { return vec2(lhs * rhs.x, lhs * rhs.y); }
	__host__ __device__ inline vec2 operator /(const vec2& lhs, const vec2& rhs) { return vec2(lhs.x / rhs.x, lhs.y / rhs.y); }
	__host__ __device__ inline vec2 operator /(const vec2& lhs, const float& rhs) { return vec2(lhs.x / rhs, lhs.y / rhs); }

	__host__ __device__ inline vec2& operator +=(vec2& lhs, const vec2& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; ; return lhs; }
	__host__ __device__ inline vec2& operator -=(vec2& lhs, const vec2& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y;  return lhs; }
	__host__ __device__ inline vec2& operator *=(vec2& lhs, const float& rhs) { lhs.x *= rhs; lhs.y *= rhs; return lhs; }
	__host__ __device__ inline vec2& operator /=(vec2& lhs, const float& rhs) { lhs.x /= rhs; lhs.y /= rhs; return lhs; }

	template<int ActualSize, int X, int Y>
	__host__ __device__ inline  __vec_swizzle<float, ActualSize, 2, X, Y>& operator +=(__vec_swizzle<float, ActualSize, 2, X, Y>& lhs, const vec2& rhs)
	{
		lhs.data[X] += rhs.x; lhs.data[Y] += rhs.y; return lhs;
	}
	template<int ActualSize, int X, int Y>
	__host__ __device__ inline  __vec_swizzle<float, ActualSize, 2, X, Y>& operator -=(__vec_swizzle<float, ActualSize, 2, X, Y>& lhs, const vec2& rhs)
	{
		lhs.data[X] -= rhs.x; lhs.data[Y] -= rhs.y; return lhs;
	}

	__host__ __device__ inline float dot(const vec2& lhs, const vec2& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }
	__host__ __device__ inline vec2 perpendicular(const vec2& lhs) { return vec2(-lhs.y, lhs.x); }
	__host__ __device__ inline float length2(const vec2& v) { return v.x * v.x + v.y * v.y; }
	__host__ __device__ inline float length(const vec2& v) { return sqrt(v.x * v.x + v.y * v.y); }
	__host__ __device__ inline vec2 normalize(const vec2& v) { return v / length(v); }
	__host__ __device__ inline vec2 fmod(const vec2& a, const vec2& b) { return vec2(fmodf(a.x, b.x), fmodf(a.y, b.y)); }
	__host__ __device__ inline vec2 fmod(const vec2& a, const float& b) { return vec2(fmodf(a.x, b), fmodf(a.y, b)); }
	__host__ __device__ inline vec2 pow(const vec2& a, const vec2& b) { return vec2(powf(a.x, b.x), powf(a.y, b.y)); }
	__host__ __device__ inline vec2 exp(const vec2& a) { return vec2(expf(a.x), expf(a.y)); }
	__host__ __device__ inline vec2 log(const vec2& a) { return vec2(logf(a.x), logf(a.y)); }
	__host__ __device__ inline vec2 log10(const vec2& a) { return vec2(log10f(a.x), log10f(a.y)); }
	__host__ __device__ inline vec2 log2(const vec2& a) { return vec2(log2f(a.x), log2f(a.y)); }
	__host__ __device__ inline vec2 clamp(const vec2& v, const vec2& a, const vec2& b) { return vec2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
	__host__ __device__ inline vec2 saturate(const vec2& v, const vec2& a, const vec2& b) { return vec2(clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f)); }
	__host__ __device__ inline vec2 abs(const vec2& a) { return vec2(fabs(a.x), fabs(a.y)); }
	__host__ __device__ inline float sum(const vec2& a) { return a.x + a.y; }
	__host__ __device__ inline vec2 ceil(const vec2& v) { return vec2(ceilf(v.x), ceilf(v.y)); }
	__host__ __device__ inline vec2 floor(const vec2& v) { return vec2(floorf(v.x), floorf(v.y)); }
	__host__ __device__ inline vec2 sign(const vec2& v) { return vec2(sign(v.x), sign(v.y)); }

	__host__ __device__ inline float cwiseMax(const vec2& v) { return (v.x > v.y) ? v.x : v.y; }
	__host__ __device__ inline float cwiseMin(const vec2& v) { return (v.x < v.y) ? v.x : v.y; }

	__host__ __device__ inline bool operator==(const vec2& lhs, const vec2& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }
	__host__ __device__ inline bool operator!=(const vec2& lhs, const vec2& rhs) { return lhs.x != rhs.x || lhs.y != rhs.y; }

	// FIXME: Cuda intrinsics aren't working. Why is this?
	//__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }

	template<typename T>
	__host__ __device__ inline T cast(const vec2& v) { T r; r.x = v.x; r.y = v.y; return r; }

}