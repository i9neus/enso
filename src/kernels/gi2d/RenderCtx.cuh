#pragma once

#include "FwdDecl.cuh"
#include "../CudaSampler.cuh"
#include "generic/Hash.h"

using namespace Cuda;

namespace GI2D
{
	using RNG = Cuda::PseudoRNG;
	struct RenderCtx;

	class Accumulator
	{
	public:
		__device__ Accumulator() {}

		__device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) = 0;
	};
	
	struct RenderCtx
	{
		__device__ __forceinline__ RenderCtx(const uint& sIdx, 
											 const uint accumIdx, 
											 const uchar& depth, 
										     Accumulator& accum) :
			sampleIdx(sIdx),
			rng(HashOf(sIdx, uint(depth) + 9871251u, accumIdx)),
			accumulator(accum)
		{}

		uint			sampleIdx;
		RNG				rng;
		Accumulator&	accumulator;
	};
}