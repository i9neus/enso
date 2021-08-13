#pragma once

#include "vec/CudaIVec2.cuh"
#include "vec/CudaIVec3.cuh"
#include "vec/CudaIVec4.cuh"
#include "vec/CudaVec2.cuh"
#include "vec/CudaVec3.cuh"
#include "vec/CudaVec4.cuh"
#include "mat/CudaMat3.cuh"
#include "mat/CudaMat4.cuh"

namespace Cuda
{    	
	__host__ __device__ __forceinline__ float	cubrt(float a)								{ return copysignf(1.0f, a) * powf(fabs(a), 1.0f / 3.0f); }
	template<typename T> __host__ __device__ __forceinline__ T toRad(const T& deg)			{ return kTwoPi * deg / 360.0f; }
	template<typename T> __host__ __device__ __forceinline__ T toDeg(const T& rad)			{ return 360.0f * rad / kTwoPi; }
	template<typename T> __host__ __device__ __forceinline__ T sqr(const T& a)           	{ return a * a; }
	__host__ __device__ __forceinline__ int		mod2(int a, int b)							{ return ((a % b) + b) % b; }
	__host__ __device__ __forceinline__ float	mod2(float a, float b)						{ return fmodf(fmodf(a, b) + b, b); }
	__host__ __device__ __forceinline__ vec3	mod2(vec3 a, vec3 b)						{ return fmod(fmod(a, b) + b, b); }	
	__host__ __device__ __forceinline__ int		sum(ivec2 a)								{ return a.x + a.y; }
	__host__ __device__ __forceinline__ float	luminance(vec3 v)							{ return v.x * 0.17691f + v.y * 0.8124f + v.z * 0.01063f; }
	__host__ __device__ __forceinline__ float	mean(vec3 v)								{ return (v.x + v.y + v.z) / 3; }
	__host__ __device__ __forceinline__ float	sin01(float a)								{ return 0.5f * sin(a) + 0.5f; }
	__host__ __device__ __forceinline__ float	cos01(float a)								{ return 0.5f * cos(a) + 0.5f; }
	__host__ __device__ __forceinline__ float	saturate(float a)							{ return clamp(a, 0.0, 1.0); }
	__host__ __device__ __forceinline__ float	saw01(float a)								{ return fabs(fract(a) * 2 - 1); }
	__host__ __device__ __forceinline__ void	sort(float& a, float& b)					{ if(a > b) { float s = a; a = b; b = s; } }
	__host__ __device__ __forceinline__ void	swap(float& a, float& b)					{ float s = a; a = b; b = s; }
	__host__ __device__ __forceinline__ float	max3(const float& a, const float& b, const float& c) { return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c); }
	__host__ __device__ __forceinline__ float	min3(const float& a, const float& b, const float& c) { return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c); }
	template<typename T> __host__ __device__ __forceinline__ T mix(const T& a, const T& b, const float& v) { return a * (1 - v) + b * v; }
	template<typename T> __host__ __forceinline__ void echo(const T& t) { std::printf("%s\n", t.format().c_str()); }

	__host__ __device__ __forceinline__ float	Volume(const vec3& v)						{ return v.x * v.y * v.z; }
	__host__ __device__ __forceinline__ int		Volume(const ivec3& v)						{ return v.x * v.y * v.z; }
	__host__ __device__ __forceinline__ uint	Volume(const uvec3& v)						{ return v.x * v.y * v.z; }
	__host__ __device__ __forceinline__ float	Area(const vec2& v)							{ return v.x * v.y; }
	__host__ __device__ __forceinline__ int		Area(const ivec2& v)						{ return v.x * v.y; }
	__host__ __device__ __forceinline__ uint	Area(const uvec2& v)						{ return v.x * v.y; }

	__host__ __device__ __forceinline__ vec3	PolarToCartesian(const vec2& polar)
	{
		const float sinTheta = sin(polar.x);
		return vec3(sin(polar.y) * sinTheta, cos(polar.x), cos(polar.y) * sinTheta);
	}

	#define kOne vec3(1.0f)
	#define kZero vec3(0.0f)

	#define kZero4f vec4(0.0f)
	#define kZero3f vec3(0.0f)
	#define kZero2f vec2(0.0f)
	#define kZero4i ivec4(0)
	#define kZero3i ivec3(0)
	#define kZero2i ivec2(0)
	#define kZero4u uvec4(0u)
	#define kZero3u uvec3(0u)
	#define kZero2u uvec2(0u)

	#define kBlack vec3(0.0f)
	#define kWhite vec3(1.0f)
	#define kRed vec3(1.0f, 0.0f, 0.0f)
	#define kYellow vec3(1.0f, 1.0f, 0.0f)
	#define kGreen vec3(0.0f, 1.0f, 0.0f)
	#define kBlue vec3(0.0f, 0.0f, 1.0f)
	#define kPink vec3(1.0f, 0.0f, 1.0f)
}
