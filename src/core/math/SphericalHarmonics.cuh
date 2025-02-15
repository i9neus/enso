﻿#pragma once

#include "Math.cuh"

namespace Enso
{
	namespace SH
	{
		__host__ __device__ __forceinline__ float Project(const vec3& n, const int L, const int M)
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
				}
			}
			}

			printf("Invalid SH index L = %i, M = %i\n", L, M);
			CudaAssert(false);
		}

		__host__ __device__ __forceinline__ float Project(const vec3& n, const uint idx)
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
			}

			printf("Invalid SH index %u\n", idx);
			CudaAssert(false);
		}

		__host__ __device__ __forceinline__ float Legendre(const uint idx)
		{
			switch (idx)
			{
			case 0:		return 0.2820947917738781f;
			case 1:		return 0.4886025119029199f;
			case 2:		return 0.4886025119029199f;
			case 3:		return 0.4886025119029199f;
			case 4:		return 1.0925484305920792f;
			case 5:		return 1.0925484305920792f;
			case 6:		return 0.3153915652525200f;
			case 7:		return 1.0925484305920792f;
			case 8:		return 0.5462742152960396f;
			case 9:		return 0.5900435899266435f;
			case 10:	return 2.890611442640554f;
			case 11:	return 0.4570457994644658f;
			case 12:	return 0.3731763325901154f;
			case 13:	return 0.4570457994644658f;
			case 14:	return 1.445305721320277f;
			case 15:	return 0.5900435899266435f;
			}

			printf("Invalid SH index %u\n", idx);
			CudaAssert(false);
		}

		__host__ __device__ __forceinline__ float GetLegendreCoefficient(const int L, const int M)
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
				default:	CudaAssert(false);
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
				default:	CudaAssert(false);
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
				default:	CudaAssert(false);
				}
			}
			}

			CudaAssert(false);
		}

		__host__ __device__ __forceinline__ int GetNumCoefficients(const int L) { return (L + 1) * (L + 1); }
	}
}