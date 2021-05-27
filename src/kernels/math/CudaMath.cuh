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
	//__host__ __device__ template<typename T> inline T min(const T& a, const T& b) { return (a < b) ? a : b; }
	//__host__ __device__ template<typename T> inline T max(const T& a, const T& b) { return (a > b) ? a : b; }
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
	
	template<typename T>
	__host__ __device__ inline T mix(const T& a, const T& b, const float& v) { return T(float(a) * (1 - v) + float(b) * v); }

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

	// Fast construction of orthonormal basis using quarternions to avoid expensive normalisation and branching 
	// From Duf et al's technical report https://graphics.pixar.com/library/OrthonormalB/paper.pdf, inspired by
	// Frisvad's original paper: http://orbit.dtu.dk/files/126824972/onb_frisvad_jgt2012_v2.pdf
	__host__ __device__ inline mat4 createBasis(vec3 n)
	{
		float s = sign(n.z);
		float a = -1.0 / (s + n.z);
		float b = n.x * n.y * a;

		return transpose(mat4(vec4(1.0f + s * n.x * n.x * a, s * b, -s * n.x, 0.0),
			vec4(b, s + n.y * n.y * a, -n.y, 0.0),
			vec4(n, 0.0),
			vec4(0.0, 0.0, 0.0, 1.0)));
	}
	
	/*__host__ __device__ inline mat4 createBasis(vec3 n)
	{
		float s = sign(n.z);
		float a = -1.0 / (s + n.z);
		float b = n.x * n.y * a;

		return transpose(mat4(vec4(1.0f + s * n.x * n.x * a, s * b, -s * n.x, 0.0),
			vec4(b, s + n.y * n.y * a, -n.y, 0.0),
			vec4(n, 0.0),
			vec4(0.0, 0.0, 0.0, 1.0)));

		vec3 tangent = normalize(cross(n, (abs(dot(n, vec3(1.0, 0.0, 0.0))) < 0.5) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
		vec3 cotangent = cross(tangent, n);

		return transpose(mat4(vec4(tangent, 0.0), vec4(cotangent, 0.0), vec4(n, 0.0), vec4(kZero, 1.0)));
	}*/

	__host__ __device__ inline mat4 createBasis(vec3 n, vec3 up)
	{
		float s = sign(n.z);
		float a = -1.0 / (s + n.z);
		float b = n.x * n.y * a;

		return transpose(mat4(vec4(1.0f + s * n.x * n.x * a, s * b, -s * n.x, 0.0),
			vec4(b, s + n.y * n.y * a, -n.y, 0.0),
			vec4(n, 0.0),
			vec4(0.0, 0.0, 0.0, 1.0)));

		vec3 tangent = normalize(cross(n, up));
		vec3 cotangent = cross(tangent, n);

		return transpose(mat4(vec4(tangent, 0.0), vec4(cotangent, 0.0), vec4(n, 0.0), vec4(kZero, 1.0)));
	}
}
