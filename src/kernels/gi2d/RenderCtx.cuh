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
		__device__ __forceinline__ RenderCtx(const uint& probeHash, 
											 const uint accumIdx, 
											 const uchar& depth, 
										     Device::Camera2D& cam) :
			hash(HashOf(probeHash, uint(depth) + 9871251u, accumIdx)),
			rng(hash),
			camera(cam)
		{
		}

		uint			hash;
		RNG				rng;
		Device::Camera2D& camera;
	};
}