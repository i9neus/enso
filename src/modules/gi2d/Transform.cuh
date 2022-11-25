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
        __host__ __device__ BidirectionalTransform2D() : 
            trans(0.0f), rot(0.0f), scale(1.0f)
        {
            fwd = mat2::Indentity();
            inv = mat2::Indentity();
            nInv = mat2::Indentity();
        }

        __host__ __device__ void Prepare()
        {
            inv = nInv = inverse(fwd);
        }
        
        __host__ __device__  void Construct(const vec2& tr, const float r, const float sc)
        {
            trans = tr;
            rot = r;
            scale = sc;
            
            if (rot != 0.0f)
            {
                const float sinTheta = sin(rot);
                const float cosTheta = cos(rot);
                fwd.i00 = cosTheta; fwd.i01 = sinTheta;
                fwd.i10 = sinTheta; fwd.i11 = -cosTheta;
            }
            else
            {
                fwd = mat2::Indentity();
            }

            fwd[0] *= sc; fwd[1] *= sc;

            Prepare();
        }

        __host__ __device__ __inline__ RayBasic2D RayToObjectSpace(const RayBasic2D& world) const
        {
            RayBasic2D obj;
            obj.o = world.o - trans;
            obj.d = world.d + obj.o;
            obj.o = fwd * obj.o;
            obj.d = (fwd * obj.d) - obj.o;
            return obj;
        }

        __host__ __device__ __inline__ vec2 NormalToWorldSpace(const vec2& n) const
        {
            return nInv * n;
        }

        __host__ __device__ inline vec2 PointToObjectSpace(const vec2& world) const { return fwd * (world - trans); }
        __host__ __device__ inline vec2 PointToWorldSpace(const vec2& obj) const { return (inv * obj) + trans; }

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