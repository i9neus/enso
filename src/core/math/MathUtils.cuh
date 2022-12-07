#pragma once

#include "Constants.h"
#include "../CudaHeaders.cuh"

namespace Enso
{
	// saturate() causes a compiler error when called from a function with both __host__ and __device decorators. Use saturatef() instead.
#ifdef __CUDA_ARCH__
	#define saturatef(a) __saturatef(a)
#else
	inline float saturatef(const float a) { return (a < 0.0f) ? 0.0f : ((a > 1.0f) ? 1.0f : a); }
#endif	

	// Define implementations of min and max for integral types
	template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
	__host__ __device__ inline T max(const T a, const T b) { return a > b ? a : b; }
	template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
	__host__ __device__ inline T min(const T a, const T b) { return a < b ? a : b; }

	// Min/max for triples
	template<typename T>
	__host__ __device__ __forceinline__ T	max3(const T& a, const T& b, const T& c) { return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c); }
	template<typename T>
	__host__ __device__ __forceinline__ T	min3(const T& a, const T& b, const T& c) { return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c); }

	__host__ __device__ __forceinline__ float	clamp(const float& v, const float& a, const float& b) noexcept { return fmaxf(a, fminf(v, b)); }
	template<typename T> __host__ __device__ __forceinline__ T clamp(const T& v, const T& a, const T& b) noexcept { return fmaxf(a, fminf(v, b)); }
	__host__ __device__ __forceinline__ float	fract(const float& v) noexcept { return fmodf(v, 1.0f); }
	__host__ __device__ __forceinline__ float	sign(const float& v) noexcept { return copysign(1.0f, v); }	
	__host__ __device__ __forceinline__ float	cubrt(float a) { return copysignf(1.0f, a) * powf(fabs(a), 1.0f / 3.0f); }
	template<typename T> __host__ __device__ __forceinline__ T toRad(const T& deg) { return kTwoPi * deg / 360.0f; }
	template<typename T> __host__ __device__ __forceinline__ T toDeg(const T& rad) { return 360.0f * rad / kTwoPi; }
	template<typename T> __host__ __device__ __forceinline__ T sqr(const T& a) { return a * a; }
	template<typename T> __host__ __device__ __forceinline__ T cub(const T& a) { return a * a * a; }
	__host__ __device__ __forceinline__ int		mod2(int a, int b) { return ((a % b) + b) % b; }
	__host__ __device__ __forceinline__ float	mod2(float a, float b) { return fmodf(fmodf(a, b) + b, b); }
	__host__ __device__ __forceinline__ float	sin01(float a) { return 0.5f * sin(a) + 0.5f; }
	__host__ __device__ __forceinline__ float	cos01(float a) { return 0.5f * cos(a) + 0.5f; }
	__host__ __device__ __forceinline__ float	saw01(float a) { return fabs(fract(a) * 2 - 1); }
	__host__ __device__ __forceinline__ void	sort(float& a, float& b) { if (a > b) { float s = a; a = b; b = s; } }
	__host__ __device__ __forceinline__ void	swap(float& a, float& b) { float s = a; a = b; b = s; }	
	template<typename A, typename B, typename V> __host__ __device__ __forceinline__ A mix(const A& a, const B& b, const V& v) { return a * (V(1) - v) + b * v; }
	template<typename T> __host__ __forceinline__ void echo(const T& t) { std::printf("%s\n", t.format().c_str()); }

	__host__ __device__ __forceinline__ float SignedLog(const float& t) { return logf(1.0f + fabs(t)) * copysignf(1.0f, t); }
	__host__ __device__ __forceinline__ float SignedLog10(const float& t) { return log10f(1.0f + fabs(t)) * copysignf(1.0f, t); }
	__host__ __device__ __forceinline__ float SignedLog2(const float& t) { return log2f(1.0f + fabs(t)) * copysignf(1.0f, t); }

	template<typename T>
	__host__ __device__ __forceinline__ T NearestPow2Ceil(const T& j)
	{
		return (int(j) <= 1) ? T(1) : T(1 << int(std::ceil(std::log2(float(j)))));
	}

	template<typename T>
	__host__ __device__ __forceinline__ T NearestPow2Floor(const T& j)
	{
		return  (int(j) <= 2) ? T(1) : T(1 << int(std::floor(std::log2(float(j)))));
	}

	__host__ __device__ __forceinline__ float IntToUnitFloat(const uint& integer)
	{
		return float(integer) / float(0xffffffff);
	}

	__host__ __device__ __forceinline__ float IntToUnitFloat(const int& integer)
	{
		return float(integer) / float(0x7fffffff);
	}
}
