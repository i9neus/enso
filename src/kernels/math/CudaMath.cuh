#pragma once

#include "CudaTransform.cuh"

namespace Cuda
{    	
	__host__ void VerifyTypeSizes();
	
	__host__ __device__ inline float cubrt(float a)								{ return copysignf(1.0f, a) * powf(fabs(a), 1.0f / 3.0f); }
	__host__ __device__ inline float toRad(float deg)								{ return kTwoPi * deg / 360; }
	__host__ __device__ inline float toDeg(float rad)								{ return 360 * rad / kTwoPi; }
	template<typename T> __host__ __device__ inline T sqr(const T& a)           	{ return a * a; }
	__host__ __device__ inline int mod2(int a, int b)								{ return ((a % b) + b) % b; }
	__host__ __device__ inline float mod2(float a, float b)							{ return fmodf(fmodf(a, b) + b, b); }
	__host__ __device__ inline vec3 mod2(vec3 a, vec3 b)							{ return fmod(fmod(a, b) + b, b); }	
	__host__ __device__ inline int sum(ivec2 a)										{ return a.x + a.y; }
	__host__ __device__ inline float luminance(vec3 v)								{ return v.x * 0.17691f + v.y * 0.8124f + v.z * 0.01063f; }
	__host__ __device__ inline float mean(vec3 v)									{ return v.x / 3 + v.y / 3 + v.z / 3; }
	__host__ __device__ inline float sin01(float a)								{ return 0.5f * sin(a) + 0.5f; }
	__host__ __device__ inline float cos01(float a)								{ return 0.5f * cos(a) + 0.5f; }
	__host__ __device__ inline float saturate(float a)								{ return clamp(a, 0.0, 1.0); }
	__host__ __device__ inline float saw01(float a)								{ return fabs(fract(a) * 2 - 1); }
	__host__ __device__ inline void sort(float& a, float& b)						{ if(a > b) { float s = a; a = b; b = s; } }
	__host__ __device__ inline void swap(float& a, float& b)						{ float s = a; a = b; b = s; }
	__host__ __device__ inline float max3(const float& a, const float& b, const float& c) { return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c); }
	__host__ __device__ inline float min3(const float& a, const float& b, const float& c) { return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c); }
	template<typename T> __host__ __device__ inline T mix(const T& a, const T& b, const float& v) { return T(float(a) * (1 - v) + float(b) * v); }
	template<typename T> __host__ inline void echo(const T& t) { std::printf("%s\n", t.format().c_str()); }

	#define KERNEL_COORDS_IVEC2 ivec2(blockIdx.x* blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)	

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

	// Finds the roots of a quadratic equation of the form a.x^2 + b.x + c = 0
	__device__ inline bool QuadraticSolve(float a, float b, float c, float& t0, float& t1)
	{
		float b2ac4 = b * b - 4 * a * c;
		if (b2ac4 < 0) { return false; }

		float sqrtb2ac4 = sqrt(b2ac4);
		t0 = (-b + sqrtb2ac4) / (2 * a);
		t1 = (-b - sqrtb2ac4) / (2 * a);
		return true;
	}
}
