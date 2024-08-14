#pragma once

#include "Mat2.cuh"

namespace Enso
{
	// 1x2 * 2x2 = 2x1
	__host__ __device__ __forceinline__ vec2 operator*(const vec2& a, const mat2& B)
	{
		return vec2(a.x * B.i00 + a.y * B.i01, a.x * B.i10 + a.y * B.i11);
	}
}