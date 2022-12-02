#include "Transform2D.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __host__ __device__ BidirectionalTransform2D::BidirectionalTransform2D() :
        trans(0.0f), rot(0.0f), scale(1.0f)
    {
        fwd = mat2::Identity();
        inv = mat2::Identity();
        nInv = mat2::Identity();
    }

    __host__ __device__ void BidirectionalTransform2D::Prepare()
    {
        inv = nInv = inverse(fwd);
    }

    __host__ __device__ void BidirectionalTransform2D::Construct(const vec2& tr, const float r, const float sc)
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
            fwd = mat2::Identity();
        }

        fwd[0] *= sc; fwd[1] *= sc;

        Prepare();
    }
    
    __host__ bool BidirectionalTransform2D::Serialise(Json::Node& node, const int flags) const
    {
        Json::Node transNode = node.AddChildObject("transform");
        
        transNode.AddVector("tra", trans);
        transNode.AddValue("rot", rot);
        transNode.AddVector("sca", scale);
        return true;
    }

    __host__ bool BidirectionalTransform2D::Deserialise(const Json::Node& node, const int flags)
    {
        const Json::Node transNode = node.GetChildObject("transform", flags);
        if (!transNode) { return false; }
        
        transNode.GetVector("tra", trans, flags);
        transNode.GetValue("rot", rot, flags);
        transNode.GetVector("sca", scale, flags);
        return true;
    }

    __host__ __device__ ViewTransform2D::ViewTransform2D()
    {
        matrix = mat3::Identity();
        scale = 1.f;
        rotate = 0.f;
        trans = vec2(0.f);
    }

    __host__ __device__ ViewTransform2D::ViewTransform2D(const mat3& mat, const vec2& tra, const float& rot, const float& sca) :
        matrix(mat), trans(tra), rotate(rot), scale(sca) {}

    __host__ mat3 ConstructViewMatrix(const vec2& trans, const float rotate, const float scale)
    {
        const float sinTheta = std::sin(rotate);
        const float cosTheta = std::cos(rotate);
        mat3 m = mat3::Identity();
        m.i00 = scale * cosTheta; m.i01 = scale * sinTheta;
        m.i10 = scale * sinTheta; m.i11 = scale * -cosTheta;
        m.i02 = trans.x;
        m.i12 = trans.y;
        return m;
    }
}
