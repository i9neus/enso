#pragma once

#include "Ray2D.cuh"

using namespace Cuda;

namespace Cuda
{
    namespace Host { template<typename T> class Vector; }
}

namespace GI2D
{
	class BidirectionalTransform2D
	{
	public:
		vec2 trans;
		float rot;
		vec2 scale;

		mat2 fwd;
		mat2 inv;
		mat2 nInv;
	};
}