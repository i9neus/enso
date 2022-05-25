#include "CudaTransform.cuh"
#include "generic/JsonUtils.h"

#include <random>

namespace Cuda
{
    __host__ __device__ BidirectionalTransform::BidirectionalTransform()
    {
        trans = vec3(0.0f);
        rot = vec3(0.0f);
        scale = vec3(1.0f);
        
        MakeIdentity();
    }

    __host__ void BidirectionalTransform::Update(const uint operation)
    {
        trans.Update(operation);
        rot.Update(operation);
        scale.Update(operation);

        scale.eval.y = scale.eval.z = scale.eval.x;
        scale.t.y = scale.t.z = scale.t.x;
    }
    
    __host__ uint BidirectionalTransform::FromJson(const ::Json::Node& node, const uint flags)
    {
        const auto transNode = node.GetChildObject("transform", flags);
        if (!transNode) { return 0u; }        

        trans.FromJson("pos", transNode, flags);
        rot.FromJson("rot", transNode, ::Json::kSilent);
        scale.FromJson("sca", transNode, ::Json::kSilent);
        
        Rebuild();
        return 0u;
    }

    __host__ __device__ void BidirectionalTransform::Set(const vec3& t, const vec3& r, const vec3& s)
    {
        trans = t;
        rot = r;
        scale = s;

        Rebuild();
    }

    __host__ __device__ void BidirectionalTransform::MakeScaleUniform()
    {
        scale.eval.y = scale.eval.z = scale.eval.x;
        scale.t.y = scale.t.z = scale.t.x;

        Rebuild();
    }

    __host__ void BidirectionalTransform::ToJson(Json::Node& parentNode) const
    {
        auto transNode = parentNode.AddChildObject("transform");

        trans.ToJson("pos", transNode);
        rot.ToJson("rot", transNode);
        scale.ToJson("sca", transNode);
    }

    __host__ BidirectionalTransform::BidirectionalTransform(const ::Json::Node& node, const uint flags)
    {
        FromJson(node, flags);
    }

    __host__ __device__ void BidirectionalTransform::Rebuild()
    {
        fwd = mat3::Indentity();

        if (rot().x != 0.0) { fwd *= RotXMat3(toRad(rot().x)); }
        if (rot().y != 0.0) { fwd *= RotYMat3(toRad(rot().y)); }
        if (rot().z != 0.0) { fwd *= RotZMat3(toRad(rot().z)); }

        nInv = transpose(fwd);

        if (scale() != vec3(1.0f)) { fwd *= ScaleMat3(scale()); }

        inv = inverse(fwd);
    }
}