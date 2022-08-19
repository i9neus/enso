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

    struct ViewTransform2D
    {
        __host__ __device__ ViewTransform2D()
        {
            matrix = mat3::Indentity();
            scale = 1.f;
            rotate = 0.f;
            trans = vec2(0.f);
            mouseView = 0.0f;
            dPdXY = 1.0 / 1024.f;
        }

        __host__ __device__ ViewTransform2D(const mat3& mat, const vec2& tra, const float& rot, const float& sca, const vec2& mv, const float& dp) :
            matrix(mat), trans(tra), rotate(rot), scale(sca), mouseView(mv), dPdXY(dp) {}

        mat3 matrix;
        vec2 trans;
        float rotate;
        float scale;

        vec2 mouseView;
        float dPdXY;
    };
}