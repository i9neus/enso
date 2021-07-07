#pragma once

#include "CudaTransform.cuh"

namespace Cuda
{    	
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
		float		sqrBoundRadius;
	};
}
