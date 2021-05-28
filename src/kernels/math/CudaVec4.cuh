#pragma once

#include "../CudaCommonIncludes.cuh"
#include "generic/Constants.h"
#include "CudaVec3.cuh"

namespace Cuda
{	
	template<int T, int S, int A, int B, int C, int D>
	struct VecSwiz
	{
		float data[S];
	};
	
	struct __align__(16) vec4 : public VecBase<4>
	{
		enum _attrs : size_t { kDims = 4 };
		using kType = float;

		union
		{
			struct { float x, y, z, w; };
			struct { float i0, i1, i2, i3; };
			struct { vec3 xyz; float _0_w; };
			struct { vec2 xy; vec2 zw; };
			float data[4];
		};

		vec4() = default;
		__host__ __device__ explicit vec4(const float v) : x(v), y(v), z(v), w(v) {}
		__host__ __device__ vec4(const float& x_, const float& y_, const float& z_, const float& w_) : x(x_), y(y_), z(z_), w(w_) {}
		__host__ __device__ vec4(const vec3& v, const float& w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
		__host__ __device__ vec4(const vec2& v, const float& z_, const float& w_) : x(v.x), y(v.y), z(z_), w(w_) {}
		__host__ __device__ vec4(const vec4 & other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
		template<typename T, typename = std::enable_if<std::is_base_of<VecBase<4>, T>::value>::type> 
		__host__ __device__ vec4(const T& other) : x(float(other.x)), y(float(other.y)), z(float(other.z)), w(float(other.w)) {}

		__host__ __device__ vec4& operator=(const float& v) { x = v; y = v; z = v; w = v; return *this; }

		__host__ __device__ inline const float& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline float& operator[](const unsigned int idx) { return data[idx]; }
		__host__ __device__ inline vec4& operator=(const float v) { x = y = z = w = v;  return *this; }

		__host__ inline std::string format() const { return tfm::format("{%.10f, %.10f, %.10f, %.10f}", x, y, z, w); }
	};

	__host__ __device__ inline vec4 operator +(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
	__host__ __device__ inline vec4 operator -(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
	__host__ __device__ inline vec4 operator -(const vec4& lhs) { return vec4(-lhs.x, -lhs.y, -lhs.z, -lhs.w); }
	__host__ __device__ inline vec4 operator *(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }
	__host__ __device__ inline vec4 operator *(const vec4& lhs, const float& rhs) { return vec4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs); }
	__host__ __device__ inline vec4 operator *(const float& lhs, const vec4& rhs) { return vec4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w); }
	__host__ __device__ inline vec4 operator /(const vec4& lhs, const float& rhs) { return vec4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs); }
	__host__ __device__ inline vec4& operator +=(vec4& lhs, const vec4& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs; }
	__host__ __device__ inline vec4& operator -=(vec4& lhs, const vec4& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs; }
	__host__ __device__ inline vec4& operator *=(vec4& lhs, const float& rhs) { lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; lhs.w *= rhs; return lhs; }
	__host__ __device__ inline vec4& operator /=(vec4& lhs, const float& rhs) { lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs; lhs.w /= rhs; return lhs; }

	__host__ __device__ inline float dot(const vec4& lhs, const vec4& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w; }
	__host__ __device__ inline float length2(const vec4& v) { return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
	__host__ __device__ inline float length(const vec4& v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w); }
	__host__ __device__ inline vec4 normalize(const vec4& v) { return v / length(v); }
	__host__ __device__ inline vec4 fmod(const vec4& a, const vec4& b) { return vec4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w)); }
	__host__ __device__ inline vec4 pow(const vec4& a, const vec4& b) { return vec4(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z), powf(a.w, b.w)); }
	__host__ __device__ inline vec4 exp(const vec4& a) { return vec4(expf(a.x), expf(a.y), expf(a.z), expf(a.w)); }
	__host__ __device__ inline vec4 log(const vec4& a) { return vec4(logf(a.x), logf(a.y), logf(a.z), logf(a.w)); }
	__host__ __device__ inline vec4 log10(const vec4& a) { return vec4(log10f(a.x), log10f(a.y), log10f(a.z), log10f(a.w)); }
	__host__ __device__ inline vec4 log2(const vec4& a) { return vec4(log2f(a.x), log2f(a.y), log2f(a.z), log2f(a.w)); }
	__host__ __device__ inline vec4 clamp(const vec4& v, const vec4& a, const vec4& b) { return vec4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }
	__host__ __device__ inline vec4 saturate(const vec4& v, const vec4& a, const vec4& b) { return vec4(clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f), clamp(v.z, 0.0f, 1.0f), clamp(v.w, 0.0f, 1.0f)); }
	__host__ __device__ inline vec4 abs(const vec4& a) { return vec4(fabs(a.x), fabs(a.y), fabs(a.z), fabs(a.w)); }
	__host__ __device__ inline float sum(const vec4& a) { return a.x + a.y + a.z + a.w; }

	__host__ __device__ inline float cwiseMax(const vec4& v) 
	{ 
		float m = -std::numeric_limits<float>::max();
		for (int i = 0; i < 4; i++) { m = fmax(m, v[i]); }
		return m;
	}
	__host__ __device__ inline float cwiseMin(const vec4& v)
	{
		float m = std::numeric_limits<float>::max();
		for (int i = 0; i < 4; i++) { m = fmin(m, v[i]); }
		return m;
	}

	// FIXME: Cuda intrinsics aren't working. Why is this?
	//__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }

	template<typename T>
	__host__ __device__ inline T cast(const vec4& v) { T r; r.x = v.x; r.y = v.y; r.z = v.z; r.w = v.w; return r; }
}