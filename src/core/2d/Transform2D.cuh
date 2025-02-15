#pragma once

#include "Ray2D.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }
    namespace Json { class Node; }
    
    class BidirectionalTransform2D
	{
	public:
        __host__ __device__ BidirectionalTransform2D();

        __host__ __device__ void Prepare();        
        __host__ __device__ void Construct(const vec2& tr, const float r, const float sc);

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

        __host__ __device__ inline vec2 PointToObjectSpace(const vec2& world) const
        {
            return fwd * (world - trans);
        }

        __host__ __device__ inline vec2 PointToWorldSpace(const vec2& obj) const
        {
            return (inv * obj) + trans;
        }

        __host__ bool Serialise(Json::Node& rootNode, const int flags) const;
        __host__ bool Deserialise(const Json::Node& rootNode, const int flags);

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
        __host__ __device__ ViewTransform2D();
        __host__ __device__ ViewTransform2D(const mat3& mat, const vec2& tra, const float& rot, const float& sca);

        mat3 matrix;
        vec2 trans;
        float rotate;
        float scale;
    }; 

    __host__ mat3 ConstructViewMatrix(const vec2& trans, const float rotate, const float scale);    

    // Maps the logical pixel to to view space. The final range is:
    // [ [-viewportRes.x/viewportRes.y, -1], [viewportRes.x/viewportRes.y, 1] ] 
    __host__ __device__ __forceinline__ vec2 TransformPixelToView(const vec2& p, const vec2& viewportRes)
    {

        return 2.f * (p - viewportRes * 0.5f) / float(viewportRes.y);
    }

    __host__ __device__ __forceinline__ vec2 TransformViewToPixel(const vec2& p, const vec2& viewportRes)
    {
        return (p * float(viewportRes.y) * 0.5f) + viewportRes * 0.5f;
    }

    // Maps from normalised screen space in the range [0, 1) to [ [-viewportRes.x/viewportRes.y, -1], [viewportRes.x/viewportRes.y, 1] ] 
    __host__ __device__ __forceinline__ vec2 TransformNormalisedScreenToView(const vec2& p, const vec2& viewportRes)
    {
        return 2 * (p - 0.5) * vec2(viewportRes.x / viewportRes.y, 1.);
    }

}