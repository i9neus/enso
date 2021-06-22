#pragma once

#include "CudaTransform.cuh"

namespace Cuda
{    	
	__host__ __device__ __forceinline__ float cubrt(float a)								{ return copysignf(1.0f, a) * powf(fabs(a), 1.0f / 3.0f); }
	__host__ __device__ __forceinline__ float toRad(float deg)								{ return kTwoPi * deg / 360; }
	__host__ __device__ __forceinline__ float toDeg(float rad)								{ return 360 * rad / kTwoPi; }
	template<typename T> __host__ __device__ __forceinline__ T sqr(const T& a)           	{ return a * a; }
	__host__ __device__ __forceinline__ int mod2(int a, int b)								{ return ((a % b) + b) % b; }
	__host__ __device__ __forceinline__ float mod2(float a, float b)							{ return fmodf(fmodf(a, b) + b, b); }
	__host__ __device__ __forceinline__ vec3 mod2(vec3 a, vec3 b)							{ return fmod(fmod(a, b) + b, b); }	
	__host__ __device__ __forceinline__ int sum(ivec2 a)										{ return a.x + a.y; }
	__host__ __device__ __forceinline__ float luminance(vec3 v)								{ return v.x * 0.17691f + v.y * 0.8124f + v.z * 0.01063f; }
	__host__ __device__ __forceinline__ float mean(vec3 v)									{ return v.x / 3 + v.y / 3 + v.z / 3; }
	__host__ __device__ __forceinline__ float sin01(float a)								{ return 0.5f * sin(a) + 0.5f; }
	__host__ __device__ __forceinline__ float cos01(float a)								{ return 0.5f * cos(a) + 0.5f; }
	__host__ __device__ __forceinline__ float saturate(float a)								{ return clamp(a, 0.0, 1.0); }
	__host__ __device__ __forceinline__ float saw01(float a)								{ return fabs(fract(a) * 2 - 1); }
	__host__ __device__ __forceinline__ void sort(float& a, float& b)						{ if(a > b) { float s = a; a = b; b = s; } }
	__host__ __device__ __forceinline__ void swap(float& a, float& b)						{ float s = a; a = b; b = s; }
	__host__ __device__ __forceinline__ float max3(const float& a, const float& b, const float& c) { return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c); }
	__host__ __device__ __forceinline__ float min3(const float& a, const float& b, const float& c) { return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c); }
	template<typename T> __host__ __device__ __forceinline__ T mix(const T& a, const T& b, const float& v) { return T(float(a) * (1 - v) + float(b) * v); }
	template<typename T> __host__ __forceinline__ void echo(const T& t) { std::printf("%s\n", t.format().c_str()); }

	#define kKernelX				(blockIdx.x * blockDim.x + threadIdx.x)	
	#define kKernelY				(blockIdx.y * blockDim.y + threadIdx.y)	
	#define kKernelIdx				kKernelX
	#define kThreadIdx				threadIdx.x
	#define kWarpLane				(threadIdx.x & 31)
	#define kKernelWidth			(gridDim.x * blockDim.x)
	#define kKernelHeight			(gridDim.y * blockDim.y)
	#define kIsFirstThread			(threadIdx.x == 0 && threadIdx.y == 0)
	
	template<typename T> __device__ __forceinline__ T kKernelPos() { return T(typename T::kType(kKernelX), typename T::kType(kKernelY)); }
	template<typename T> __device__ __forceinline__ T kKernelDims() { return T(typename T::kType(kKernelWidth), typename T::kType(kKernelHeight)); }

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

	enum class IntegratorMode { kMIS = 0, kLightOnly, kBrdfOnly };

	// Finds the roots of a quadratic equation of the form a.x^2 + b.x + c = 0
	__device__ __forceinline__ bool QuadraticSolve(float a, float b, float c, float& t0, float& t1)
	{
		float b2ac4 = b * b - 4 * a * c;
		if (b2ac4 < 0) { return false; }

		float sqrtb2ac4 = sqrt(b2ac4);
		t0 = (-b + sqrtb2ac4) / (2 * a);
		t1 = (-b - sqrtb2ac4) / (2 * a);
		return true;
	}

	template<int NumVertices, int NumFaces, int PolyOrder, typename IdxType = uchar>
	struct SimplePolyhedron
	{
		enum _attrs : int { kNumVertices = NumVertices, kNumFaces = NumFaces, kPolyOrder = PolyOrder };
		
		SimplePolyhedron() = default;
		
		vec3		V[NumVertices];
		IdxType		F[NumFaces * PolyOrder];		
	};
}
