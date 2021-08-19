#include "CudaTransform.cuh"
#include "generic/JsonUtils.h"

#include <random>

namespace Cuda
{
    __host__ void TransformParams::FromJson(const ::Json::Node& node, const uint flags, const std::string& id)
    {
        const auto subNode = node.GetChildObject(id, flags);
        if (!subNode) { return; }

        trans = rot = 0.0f;
        scale = 1.0f;

        subNode.GetVector("pos", trans, flags);
        subNode.GetVector("rot", rot, flags);
        subNode.GetVector("sca", scale, flags);
    }

    __host__ void TransformParams::ToJson(Json::Node& parentNode, const std::string& id) const
    {
        auto subNode = parentNode.AddChildObject(id);

        subNode.AddArray("pos", std::vector<float>({ trans.x, trans.y, trans.z }));
        subNode.AddArray("rot", std::vector<float>({ rot.x, rot.y, rot.z }));
        subNode.AddArray("sca", std::vector<float>({ scale.x, scale.y, scale.z }));
    }

    __host__ __device__ void TransformParams::MakeIdentity()
    {
        trans = rot = 0.0f;
        scale = 1.0f;
    }

    __host__ __device__ void TransformParams::Zero()
    {
        trans = rot = scale = 0.0f;
    }

    __host__ __device__ BidirectionalTransform::BidirectionalTransform()
    {
        p.MakeIdentity();
        dpdt.Zero();
        t.Zero();
        
        MakeIdentity();
    }

    __host__ void BidirectionalTransform::Randomise(const float xi0, const float xi1)
    {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<> rng(xi0, xi1);

        t.trans = vec3(rng(mt), rng(mt), rng(mt));
        t.rot = vec3(rng(mt), rng(mt), rng(mt));
        t.scale = vec3(rng(mt), rng(mt), rng(mt));
    }
    
    __host__ void BidirectionalTransform::FromJson(const ::Json::Node& node, const uint flags)
    {
        const auto transNode = node.GetChildObject("transform", flags);
        if (!transNode) { return; }

        p.FromJson(transNode, flags, "p");
        dpdt.FromJson(transNode, ::Json::kSilent, "dpdt");
        t.FromJson(transNode, ::Json::kSilent, "t");

        trans = p.trans + dpdt.trans * (t.trans * 2.0f - 1.0f);
        rot = p.rot + dpdt.rot * (t.rot * 2.0f - 1.0f);
        scale = p.scale + dpdt.scale * (t.scale * 2.0f - 1.0f);
        
        // Build the transform
        Create(trans, rot, scale);
    }

    __host__ void BidirectionalTransform::ToJson(Json::Node& parentNode) const
    {
        auto transNode = parentNode.AddChildObject("transform");

        p.ToJson(transNode, "p");
        dpdt.ToJson(transNode, "dpdt");
        t.ToJson(transNode, "t");
    }

    __host__ BidirectionalTransform::BidirectionalTransform(const ::Json::Node& node, const uint flags)
    {
        FromJson(node, flags);
    }
}