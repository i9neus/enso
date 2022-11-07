#pragma once

#include "FwdDecl.cuh"
#include "../CudaSampler.cuh"
#include "generic/Hash.h"

using namespace Cuda;

namespace GI2D
{
	using RNG = Cuda::PseudoRNG;
	struct RenderCtx;
	
	struct RenderCtx
	{
		__device__ __forceinline__ RenderCtx(const uint& sIdx, 
											 const uint accumIdx, 
											 const uchar& depth, 
										     Device::Camera2D& cam) :
			sampleIdx(sIdx),
			rng(HashOf(sIdx, uint(depth) + 9871251u, accumIdx)),
			camera(cam)
		{}

		uint			sampleIdx;
		RNG				rng;
		Device::Camera2D& camera;
	};
}