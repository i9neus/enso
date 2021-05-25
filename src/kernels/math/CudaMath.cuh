#pragma once

#include "CudaIVec2.cuh"
#include "CudaIVec3.cuh"
#include "CudaIVec4.cuh"
#include "CudaVec2.cuh"
#include "CudaVec3.cuh"
#include "CudaVec4.cuh"
#include "CudaMat3.cuh"
#include "CudaMat4.cuh"
#include "CudaMat3.cuh"

namespace Cuda
{
	 __host__ __device__ inline float cubrt(float a)								{ return copysignf(1.0f, a) * powf(abs(a), 1.0f / 3.0f); }
	 __host__ __device__ inline float toRad(float deg)								{ return kTwoPi * deg / 360; }
	 __host__ __device__ inline float toDeg(float rad)								{ return 360 * rad / kTwoPi; }
	 __host__ __device__ inline float sqr(float a)									{ return a * a; }
	 __host__ __device__ inline vec3 sqr(vec3 a)									{ return a * a; }
	 __host__ __device__ inline int sqr(int a)										{ return a * a; }
	 __host__ __device__ inline int mod2(int a, int b)								{ return ((a % b) + b) % b; }
	 __host__ __device__ inline float mod2(float a, float b)						{ return fmodf(fmodf(a, b) + b, b); }
	 __host__ __device__ inline vec3 mod2(vec3 a, vec3 b)							{ return fmod(fmod(a, b) + b, b); }	 
	 __host__ __device__ inline int sum(ivec2 a)									{ return a.x + a.y; }
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
