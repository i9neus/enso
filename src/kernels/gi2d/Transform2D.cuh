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
        }

        __host__ __device__ ViewTransform2D(const mat3& mat, const vec2& tra, const float& rot, const float& sca) :
            matrix(mat), trans(tra), rotate(rot), scale(sca) {}

        mat3 matrix;
        vec2 trans;
        float rotate;
        float scale;
    }; 

    __host__ __inline__ mat3 ConstructViewMatrix(const vec2& trans, const float rotate, const float scale)
    {
        const float sinTheta = std::sin(rotate);
        const float cosTheta = std::cos(rotate);
        mat3 m = mat3::Indentity();
        m.i00 = scale * cosTheta; m.i01 = scale * sinTheta;
        m.i10 = scale * sinTheta; m.i11 = scale * -cosTheta;
        m.i02 = trans.x;
        m.i12 = trans.y;
        return m;
    }
}