#pragma once

#include "../CudaSampler.cuh"
#include "generic/Hash.h"

using namespace Cuda;

namespace Cuda
{
    namespace Host { template<typename T> class Vector; }
}

namespace GI2D
{
	using RNG = Cuda::PseudoRNG;
	
	struct RenderCtx
	{
		__device__ __forceinline__ RenderCtx(const uint& sampleIdx, const uint accumIdx, const uchar& depth) :
			rng(HashOf(sampleIdx, uint(depth) + 9871251u, accumIdx))
		{}

		RNG	    rng;
	};
}