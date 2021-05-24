#pragma once

#include "CudaCommonIncludes.cuh"
#include "generic/Constants.h"

namespace Cuda
{
	__host__ __device__ inline float clamp(const float& v, const float& a, const float& b) { return fmaxf(a, fminf(v, b)); }
	__host__ __device__ inline float fract(const float& v) { return fmodf(v, 1.0f); }

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	struct __builtin_align__(16) vec4
	{
		enum _attrs : size_t { kDims = 4 };
		
		union
		{
			struct { float x, y, z, w; };
			float data[4];
		};

		vec4() = default;
		vec4(const float v) : x(v), y(v), z(v), w(v) {}
		vec4(const float& x_, const float& y_, const float& z_, const float& w_) : x(x_), y(y_), z(z_), w(w_) {}
		vec4(const vec4 & other) : x(other.x), y(other.y), z(other.z), w(other.w) {}

		__host__ __device__ inline const float& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline float& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%f, %f, %f, %f}", x, y, z, w); }
	};

	__host__ __device__ inline vec4 operator +(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
	__host__ __device__ inline vec4 operator -(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
	__host__ __device__ inline vec4 operator -(const vec4& lhs) { return vec4(-lhs.x, -lhs.y, -lhs.z, -lhs.w); }
	__host__ __device__ inline vec4 operator *(const vec4& lhs, const vec4& rhs) { return vec4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }
	__host__ __device__ inline vec4 operator *(const vec4& lhs, const float& rhs) { return vec4(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w * rhs); }
	__host__ __device__ inline vec4 operator *(const float& lhs, const vec4& rhs) { return vec4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w); }
	__host__ __device__ inline vec4 operator /(const vec4& lhs, const float& rhs) { return vec4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs); }
	__host__ __device__ inline vec4& operator +=(vec4& lhs, const vec4& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }
	__host__ __device__ inline vec4& operator -=(vec4& lhs, const vec4& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }
	__host__ __device__ inline vec4& operator *=(vec4& lhs, const float& rhs) { lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; return lhs; }
	__host__ __device__ inline vec4& operator /=(vec4& lhs, const float& rhs) { lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs; return lhs; }

	__host__ __device__ inline float dot(const vec4& lhs, const vec4& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
	__host__ __device__ inline float length2(const vec4& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
	__host__ __device__ inline float length(const vec4& v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
	__host__ __device__ inline vec4 normalise(const vec4& v) { return v / length(v); }
	__host__ __device__ inline vec4 fmod(const vec4& a, const vec4& b) { return vec4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w)); }
	__host__ __device__ inline vec4 pow(const vec4& a, const vec4& b) { return vec4(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z), powf(a.w, b.w)); }
	__host__ __device__ inline vec4 exp(const vec4& a) { return vec4(expf(a.x), expf(a.y), expf(a.z), expf(a.w)); }
	__host__ __device__ inline vec4 log(const vec4& a) { return vec4(logf(a.x), logf(a.y), logf(a.z), logf(a.w)); }
	__host__ __device__ inline vec4 log10(const vec4& a) { return vec4(log10f(a.x), log10f(a.y), log10f(a.z), log10f(a.w)); }
	__host__ __device__ inline vec4 log2(const vec4& a) { return vec4(log2f(a.x), log2f(a.y), log2f(a.z), log2f(a.w)); }
	__host__ __device__ inline vec4 clamp(const vec4& v, const vec4& a, const vec4& b) { return vec4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }
	__host__ __device__ inline vec4 saturate(const vec4& v, const vec4& a, const vec4& b) { return vec4(clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f), clamp(v.z, 0.0f, 1.0f), clamp(v.w, 0.0f, 1.0f)); }
	// FIXME: Cuda intrinsics aren't working. Why is this?
	//__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct vec3
	{
		enum _attrs : size_t { kDims = 3 };
		
		union
		{
			struct { float x, y, z; };
			float data[3];
		};
		
		vec3() = default;
		vec3(const float v) : x(v), y(v), z(v) {}
		vec3(const float& x_, const float& y_, const float& z_) : x(x_), y(y_), z(z_) {}
		vec3(const vec3& other) : x(other.x), y(other.y), z(other.z) {}

		__host__ __device__ inline const float& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline float& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%f, %f, %f}", x, y, z); }
	};

	__host__ __device__ inline vec3 operator +(const vec3& lhs, const vec3& rhs)	{ return vec3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
	__host__ __device__ inline vec3 operator -(const vec3& lhs, const vec3& rhs)	{ return vec3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
	__host__ __device__ inline vec3 operator -(const vec3& lhs) { return vec3(-lhs.x, -lhs.y, -lhs.z); }
	__host__ __device__ inline vec3 operator *(const vec3& lhs, const vec3& rhs)	{ return vec3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
	__host__ __device__ inline vec3 operator *(const vec3& lhs, const float& rhs)	{ return vec3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs); }
	__host__ __device__ inline vec3 operator *(const float& lhs, const vec3& rhs)	{ return vec3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z); }
	__host__ __device__ inline vec3 operator /(const vec3& lhs, const float& rhs)	{ return vec3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs); }
	__host__ __device__ inline vec3& operator +=(vec3& lhs, const vec3& rhs)		{ lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }
	__host__ __device__ inline vec3& operator -=(vec3& lhs, const vec3& rhs)		{ lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }
	__host__ __device__ inline vec3& operator *=(vec3& lhs, const float& rhs)		{ lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; return lhs; }
	__host__ __device__ inline vec3& operator /=(vec3& lhs, const float& rhs)		{ lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs; return lhs; }

	__host__ __device__ inline float dot(const vec3& lhs, const vec3& rhs)			{ return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
	__host__ __device__ inline vec3 cross(const vec3& lhs, const vec3& rhs)
	{
		return vec3(lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x);
	}
	__host__ __device__ inline float length2(const vec3& v)							{ return v.x * v.x + v.y * v.y + v.z * v.z; }
	__host__ __device__ inline float length(const vec3& v)							{ return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
	__host__ __device__ inline vec3 normalise(const vec3& v)						{ return v / length(v); }
	__host__ __device__ inline vec3 fmod(const vec3& a, const vec3& b)				{ return vec3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z)); }
	__host__ __device__ inline vec3 pow(const vec3& a, const vec3& b)				{ return vec3(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z)); }
	__host__ __device__ inline vec3 exp(const vec3& a)								{ return vec3(expf(a.x), expf(a.y), expf(a.z)); }
	__host__ __device__ inline vec3 log(const vec3& a)								{ return vec3(logf(a.x), logf(a.y), logf(a.z)); }
	__host__ __device__ inline vec3 log10(const vec3& a)							{ return vec3(log10f(a.x), log10f(a.y), log10f(a.z)); }
	__host__ __device__ inline vec3 log2(const vec3& a)								{ return vec3(log2f(a.x), log2f(a.y), log2f(a.z)); }
	__host__ __device__ inline vec3 clamp(const vec3& v, const vec3& a, const vec3& b) { return vec3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }
	__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b) { return vec3(clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f), clamp(v.z, 0.0f, 1.0f)); }
	// FIXME: Cuda intrinsics aren't working. Why is this?
	//__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct __builtin_align__(8) vec2
	{
		enum _attrs : size_t { kDims = 2 };
		
		union
		{
			struct { float x, y; };
			float data[2];
		};

		vec2() = default;
		vec2(const float v) : x(v), y(v){}
		vec2(const float& x_, const float& y_) : x(x_), y(y_) {}
		vec2(const vec2 & other) : x(other.x), y(other.y) {}

		__host__ __device__ inline const float& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline float& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format() const { return tfm::format("{%f, %f}", x, y); }
	};

	__host__ __device__ inline vec2 operator +(const vec2 & lhs, const vec2 & rhs) { return vec2(lhs.x + rhs.x, lhs.y + rhs.y); }
	__host__ __device__ inline vec2 operator -(const vec2 & lhs, const vec2 & rhs) { return vec2(lhs.x - rhs.x, lhs.y - rhs.y); }
	__host__ __device__ inline vec2 operator -(const vec2 & lhs) { return vec2(-lhs.x, -lhs.y); }
	__host__ __device__ inline vec2 operator *(const vec2 & lhs, const vec2 & rhs) { return vec2(lhs.x * rhs.x, lhs.y * rhs.y); }
	__host__ __device__ inline vec2 operator *(const vec2 & lhs, const float& rhs) { return vec2(lhs.x * rhs, lhs.y * rhs); }
	__host__ __device__ inline vec2 operator *(const float& lhs, const vec2 & rhs) { return vec2(lhs * rhs.x, lhs * rhs.y); }
	__host__ __device__ inline vec2 operator /(const vec2 & lhs, const float& rhs) { return vec2(lhs.x / rhs, lhs.y / rhs); }
	__host__ __device__ inline vec2& operator +=(vec2 & lhs, const vec2 & rhs) { lhs.x += rhs.x; lhs.y += rhs.y; ; return lhs; }
	__host__ __device__ inline vec2& operator -=(vec2 & lhs, const vec2 & rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y;  return lhs; }
	__host__ __device__ inline vec2& operator *=(vec2 & lhs, const float& rhs) { lhs.x *= rhs; lhs.y *= rhs; return lhs; }
	__host__ __device__ inline vec2& operator /=(vec2 & lhs, const float& rhs) { lhs.x /= rhs; lhs.y /= rhs; return lhs; }

	__host__ __device__ inline float dot(const vec2 & lhs, const vec2 & rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }
	__host__ __device__ inline vec2 perpendicular(const vec2& lhs) { return vec2(-lhs.y, lhs.x); }
	__host__ __device__ inline float length2(const vec2 & v) { return v.x * v.x + v.y * v.y; }
	__host__ __device__ inline float length(const vec2 & v) { return sqrt(v.x * v.x + v.y * v.y); }
	__host__ __device__ inline vec2 normalise(const vec2 & v) { return v / length(v); }
	__host__ __device__ inline vec2 fmod(const vec2 & a, const vec2 & b) { return vec2(fmodf(a.x, b.x), fmodf(a.y, b.y)); }
	__host__ __device__ inline vec2 pow(const vec2 & a, const vec2 & b) { return vec2(powf(a.x, b.x), powf(a.y, b.y)); }
	__host__ __device__ inline vec2 exp(const vec2 & a) { return vec2(expf(a.x), expf(a.y)); }
	__host__ __device__ inline vec2 log(const vec2 & a) { return vec2(logf(a.x), logf(a.y)); }
	__host__ __device__ inline vec2 log10(const vec2 & a) { return vec2(log10f(a.x), log10f(a.y)); }
	__host__ __device__ inline vec2 log2(const vec2 & a) { return vec2(log2f(a.x), log2f(a.y)); }
	__host__ __device__ inline vec2 clamp(const vec2 & v, const vec2 & a, const vec2 & b) { return vec2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }
	__host__ __device__ inline vec2 saturate(const vec2 & v, const vec2 & a, const vec2 & b) { return vec2(clamp(v.x, 0.0f, 1.0f), clamp(v.y, 0.0f, 1.0f)); }
	// FIXME: Cuda intrinsics aren't working. Why is this?
	//__host__ __device__ inline vec3 saturate(const vec3& v, const vec3& a, const vec3& b)	{ return vec3(__saturatef(v.x), __saturatef(v.x), __saturatef(v.x)); }

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct __builtin_align__(16) mat4
	{
		enum _attrs : size_t { kDims = 4 };
		using VecType = vec4;

		union
		{
			struct { vec4 x, y, z, w; };
			vec4 data[4];
		};

		__host__ __device__  static mat4 indentity()
		{
			return mat4(vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.0f, 1.0f, 0.0f, 0.0f), vec4(0.0f, 0.0f, 1.0f, 0.0f), vec4(0.0f, 0.0f, 0.0f, 1.0f));
		}

		mat4() = default;
		~mat4() = default;
		mat4(const vec4 & x_, const vec4 & y_, const vec4 & z_, const vec4 & w_) : x(x_), y(y_), z(z_), w(w_) {}
		mat4(const mat4 & other) : x(other.x), y(other.y), z(other.z), w(other.w) {}

		__host__ __device__ inline const vec4& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline vec4& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format(const bool pretty = false) const
		{
			const char nl = pretty ? '\n' : ' ';
			return tfm::format("{%s,%c%s,%c%s,%c%}", x.format(), nl, y.format(), nl, z.format(), nl, w.format());
		}
	};

	__host__ __device__ inline mat4 operator *(const mat4& a, const mat4& b) 
	{  
		mat4 r;
		r[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] + a[0][3] * b[3][0];
		r[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] + a[1][3] * b[3][0];
		r[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] + a[2][3] * b[3][0];
		r[3][0] = a[3][0] * b[0][0] + a[3][1] * b[1][0] + a[3][2] * b[2][0] + a[3][3] * b[3][0];
		r[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] + a[0][3] * b[3][1];
		r[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] + a[1][3] * b[3][1];
		r[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] + a[2][3] * b[3][1];
		r[3][1] = a[3][0] * b[0][1] + a[3][1] * b[1][1] + a[3][2] * b[2][1] + a[3][3] * b[3][1];
		r[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] + a[0][3] * b[3][2];
		r[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] + a[1][3] * b[3][2];
		r[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] + a[2][3] * b[3][2];
		r[3][2] = a[3][0] * b[0][2] + a[3][1] * b[1][2] + a[3][2] * b[2][2] + a[3][3] * b[3][2];
		r[0][3] = a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] + a[0][3] * b[3][3];
		r[1][3] = a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] + a[1][3] * b[3][3];
		r[2][3] = a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] + a[2][3] * b[3][3];
		r[3][3] = a[3][0] * b[0][3] + a[3][1] * b[1][3] + a[3][2] * b[2][3] + a[3][3] * b[3][3];
		return r;
	}

	__host__ __device__ inline vec4 operator *(const mat4& a, const vec4& b)
	{
		vec4 r;
		r[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2] + a[0][3] * b[3];
		r[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2] + a[1][3] * b[3];
		r[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2] + a[2][3] * b[3];
		r[3] = a[3][0] * b[0] + a[3][1] * b[1] + a[3][2] * b[2] + a[3][3] * b[3];
		return r;
	}

	__host__ __device__ inline vec4 operator *(const mat4& a, const vec3& b)
	{
		vec4 r;
		r[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2] + a[0][3];
		r[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2] + a[1][3];
		r[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2] + a[2][3];
		r[3] = a[3][0] * b[0] + a[3][1] * b[1] + a[3][2] * b[2] + a[3][3];
		return r;
	}

	__host__ __device__ inline mat4 transpose(const mat4& m)
	{
		mat4 r;
		r[0][0] = m[0][0]; r[0][1] = m[1][0]; r[0][2] = m[2][0]; r[0][3] = m[3][0];
		r[1][0] = m[0][1]; r[1][1] = m[1][1]; r[1][2] = m[2][1]; r[1][3] = m[3][1];
		r[2][0] = m[0][2]; r[2][1] = m[1][2]; r[2][2] = m[2][2]; r[2][3] = m[3][2];
		r[3][0] = m[0][3]; r[3][1] = m[1][3]; r[3][2] = m[2][3]; r[3][3] = m[3][3];
		return r;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct mat3
	{
		enum _attrs : size_t { kDims = 3 };
		using VecType = vec3;
		
		union
		{
			struct { vec3 x, y, z; };
			vec3 data[4];
		};

		mat3() = default;
		~mat3() = default;
		mat3(const vec3& x_, const vec3& y_, const vec3& z_) : x(x_), y(y_), z(z_) {}
		mat3(const mat3& other) : x(other.x), y(other.y), z(other.z) {}

		__host__ __device__  static mat3 indentity()
		{
			return mat3(vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));
		}

		__host__ __device__  static mat3 null()
		{
			return mat3(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f));
		}

		__host__ __device__ inline const vec3& operator[](const unsigned int idx) const { return data[idx]; }
		__host__ __device__ inline vec3& operator[](const unsigned int idx) { return data[idx]; }

		__host__ inline std::string format(const bool pretty = false) const
		{
			const char nl = pretty ? '\n' : ' ';
			return tfm::format("{%s,%c%s,%c%s}", x.format(), nl, y.format(), nl, z.format());
		}
	};

	__host__ __device__ inline mat3 operator *(const mat3& a, const mat3& b)
	{
		mat3 r;
		r[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
		r[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
		r[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
		r[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
		r[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
		r[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
		r[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];
		r[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];
		r[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
		r[0][3] = a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3];
		r[1][3] = a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3];
		r[2][3] = a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3];
		return r;
	}

	__host__ __device__ inline vec3 operator *(const mat3& a, const vec3& b)
	{
		vec3 r;
		r[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2];
		r[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2];
		r[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2];
		return r;
	}

	__host__ __device__ inline mat3 operator *(const mat3& a, const float& b)
	{
		return mat3(a[0] * b, a[1] * b, a[2] * b);
	}

	__host__ __device__ inline float det(const mat3& m)
	{
		return m[0][0] * m[1][1] * m[2][2] + 
			   m[0][1] * m[1][2] * m[2][0] + 
			   m[0][2] * m[1][0] * m[2][1] -
			   m[0][2] * m[1][1] * m[2][0] - 
			   m[0][1] * m[1][0] * m[2][2] - 
			   m[0][0] * m[1][2] * m[2][1];
	}

	__host__ __device__ inline mat3 transpose(const mat3& m)
	{
		mat3 r;
		r[0][0] = m[0][0]; r[0][1] = m[1][0]; r[0][2] = m[2][0];
		r[1][0] = m[0][1]; r[1][1] = m[1][1]; r[1][2] = m[2][1];
		r[2][0] = m[0][2]; r[2][1] = m[1][2]; r[2][2] = m[2][2];
		return r;
	}

#define _det2(a, b, c, d) ((a) * (d) - (b) * (c))

	__host__ __device__ inline mat3 inverse(const mat3& m)
	{
		constexpr float kInverseEpsilon = 1e-20f;
		
		const float determinant = det(m);
		if (abs(determinant) < kInverseEpsilon) { return mat3::null(); }

		const mat3 adjugate = { { _det2(m[1][1], m[1][2], m[2][1], m[2][2]), -_det2(m[0][1], m[0][2], m[2][1], m[2][2]), _det2(m[0][1], m[0][2], m[1][1], m[1][2]) },
								{ - _det2(m[1][0], m[1][2], m[2][0], m[2][2]), _det2(m[0][0], m[0][2], m[2][0], m[2][2]), - _det2(m[0][0], m[0][2], m[1][0], m[1][2])},
								{ _det2(m[1][0], m[1][1], m[2][0], m[2][1]), -_det2(m[0][0], m[0][1], m[2][0], m[2][1]), _det2(m[0][0], m[0][1], m[1][0], m[1][1]) } };

		return adjugate * (1 / determinant);
	}

	__host__ __device__ inline bool operator ==(const mat3& a, const mat3& b)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				if (a[i][j] != b[i][j]) { return false; }
			}
		}
		return true;
	}
	__host__ __device__ inline bool operator !=(const mat3& a, const mat3& b) { return !(a == b); }

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	 template<typename T> __host__ inline void echo(const T& t)						{ std::printf("%s\n", t.format().c_str()); }
	 __host__ __device__ inline float cubrt(float a)								{ return copysignf(1.0f, a) * powf(abs(a), 1.0f / 3.0f); }
	 __host__ __device__ inline float toRad(float deg)								{ return kTwoPi * deg / 360; }
	 __host__ __device__ inline float toDeg(float rad)								{ return 360 * rad / kTwoPi; }
	 __host__ __device__ inline float sqr(float a)									{ return a * a; }
	 __host__ __device__ inline vec3 sqr(vec3 a)									{ return a * a; }
	 __host__ __device__ inline int sqr(int a)										{ return a * a; }
	 __host__ __device__ inline int mod2(int a, int b)								{ return ((a % b) + b) % b; }
	 __host__ __device__ inline float mod2(float a, float b)						{ return fmodf(fmodf(a, b) + b, b); }
	 __host__ __device__ inline vec3 mod2(vec3 a, vec3 b)							{ return fmod(fmod(a, b) + b, b); }
	//__host__ __device__ inline float length2(vec2 v)								{ return dot(v, v); }
	//__host__ __device__ inline float length2(vec3 v)								{ return dot(v, v); }
	//__host__ __device__ inline int sum(ivec2 a)									{ return a.x + a.y; }
	 __host__ __device__ inline float luminance(vec3 v)								{ return v.x * 0.17691f + v.y * 0.8124f + v.z * 0.01063f; }
	 __host__ __device__ inline float mean(vec3 v)									{ return v.x / 3 + v.y / 3 + v.z / 3; }
	//__host__ __device__ inline vec4 mul4(vec3 a, mat4 m)							{ return vec4(a, 1.0) * m; }
	//__host__ __device__ inline vec3 mul3(vec3 a, mat4 m)							{ return (vec4(a, 1.0) * m).xyz; }
	 __host__ __device__ inline float sin01(float a)								{ return 0.5f * sin(a) + 0.5f; }
	 __host__ __device__ inline float cos01(float a)								{ return 0.5f * cos(a) + 0.5f; }
	 __host__ __device__ inline float saturate(float a)								{ return clamp(a, 0.0, 1.0); }
	 __host__ __device__ inline float saw01(float a)								{ return abs(fract(a) * 2 - 1); }
	 __host__ __device__ inline float cwiseMax(vec3 v)								{ return (v.x > v.y) ? ((v.x > v.z) ? v.x : v.z) : ((v.y > v.z) ? v.y : v.z); }
	 __host__ __device__ inline float cwiseMin(vec3 v)								{ return (v.x < v.y) ? ((v.x < v.z) ? v.x : v.z) : ((v.y < v.z) ? v.y : v.z); }
	 __host__ __device__ inline void sort(float& a, float& b)						{ if(a > b) { float s = a; a = b; b = s; } }
	 __host__ __device__ inline void swap(float& a, float& b)						{ float s = a; a = b; b = s; }
	 __host__ __device__ inline float max3(const float& a, const float& b, const float& c) { return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c); }
	 __host__ __device__ inline float min3(const float& a, const float& b, const float& c) { return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c); }
}
