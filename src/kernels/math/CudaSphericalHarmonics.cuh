#pragma once

#include "CudaMath.cuh"

namespace Cuda
{
	namespace SH
	{
		__device__ float Project(const vec3& n, const int L, const int M)
		{
			switch (L)
			{
			case 0:
				return 0.2820947917738781f;
			case 1:
			{
				switch (M)
				{
				case -1:	return 0.4886025119029199f * n.y;
				case 0:		return 0.4886025119029199f * n.z;
				case 1:		return 0.4886025119029199f * n.x;
				default:	assert(false);
				}
			}
			case 2:
			{
				switch (M)
				{
				case -2:	return 1.0925484305920792f * n.x * n.y;
				case -1:	return 1.0925484305920792f * n.y * n.z;
				case 0:		return 0.3153915652525200f * (-sqr(n.x) - sqr(n.y) + 2 * sqr(n.z));
				case 1:		return 1.0925484305920792f * n.z * n.x;
				case 2:		return 0.5462742152960396f * (sqr(n.x) - sqr(n.y));
				default:	assert(false);
				}
			}
			case 3:
			{
				switch (M)
				{
				case -3: return 0.5900435899266435f * (3 * sqr(n.x) - sqr(n.y)) * n.y;
				case -2: return 2.890611442640554f * n.x * n.y * n.z;
				case -1: return 0.4570457994644658f * n.y * (4 * sqr(n.z) - sqr(n.x) - sqr(n.y));
				case 0: return 0.3731763325901154f * n.z * (2 * sqr(n.z) - 3 * sqr(n.x) - 3 * sqr(n.y));
				case 1: return 0.4570457994644658f * n.x * (4 * sqr(n.z) - sqr(n.x) - sqr(n.y));
				case 2: return 1.445305721320277f * (sqr(n.x) - sqr(n.y)) * n.z;
				case 3: return 0.5900435899266435f * (sqr(n.x) - 4 * sqr(n.y)) * n.x;
				default:	assert(false);
				}
			}
			}

			assert(false);
		}

		__device__ float Project(const vec3& n, const int idx)
		{
			switch (idx)
			{
			case 0:		return 0.2820947917738781f;
			case 1:		return 0.4886025119029199f * n.y;
			case 2:		return 0.4886025119029199f * n.z;
			case 3:		return 0.4886025119029199f * n.x;
			case 4:		return 1.0925484305920792f * n.x * n.y;
			case 5:		return 1.0925484305920792f * n.y * n.z;
			case 6:		return 0.3153915652525200f * (-sqr(n.x) - sqr(n.y) + 2 * sqr(n.z));
			case 7:		return 1.0925484305920792f * n.z * n.x;
			case 8:		return 0.5462742152960396f * (sqr(n.x) - sqr(n.y));
			case 9:		return 0.5900435899266435f * (3 * sqr(n.x) - sqr(n.y)) * n.y;
			case 10:	return 2.890611442640554f * n.x * n.y * n.z;
			case 11:	return 0.4570457994644658f * n.y * (4 * sqr(n.z) - sqr(n.x) - sqr(n.y));
			case 12:	return 0.3731763325901154f * n.z * (2 * sqr(n.z) - 3 * sqr(n.x) - 3 * sqr(n.y));
			case 13:	return 0.4570457994644658f * n.x * (4 * sqr(n.z) - sqr(n.x) - sqr(n.y));
			case 14:	return 1.445305721320277f * (sqr(n.x) - sqr(n.y)) * n.z;
			case 15:	return 0.5900435899266435f * (sqr(n.x) - 4 * sqr(n.y)) * n.x;
			default:	assert(false);
			}
		}

		__device__ float GetLegendreCoefficient(const int L, const int M)
		{
			switch (L)
			{
			case 0:
				return 0.2820947917738781f;
			case 1:
			{
				switch (M)
				{
				case -1:	return 0.4886025119029199f;
				case 0:		return 0.4886025119029199f;
				case 1:		return 0.4886025119029199f;
				default:	assert(false);
				}
			}
			case 2:
			{
				switch (M)
				{
				case -2:	return 1.0925484305920792f;
				case -1:	return 1.0925484305920792f;
				case 0:		return 0.3153915652525200f;
				case 1:		return 1.0925484305920792f;
				case 2:		return 0.5462742152960396f;
				default:	assert(false);
				}
			}
			case 3:
			{
				switch (M)
				{
				case -3: return 0.5900435899266435f;
				case -2: return 2.890611442640554f;
				case -1: return 0.4570457994644658f;
				case 0: return 0.3731763325901154f;
				case 1: return 0.4570457994644658f;
				case 2: return 1.445305721320277f;
				case 3: return 0.5900435899266435f;
				default:	assert(false);
				}
			}
			}

			assert(false);
		}

		__device__ __forceinline__ int GetNumCoefficients(const int L) { return (L + 1) * (L + 1); }
	}
}