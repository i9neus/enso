#include "CudaTransform.cuh"
#include "generic/JsonUtils.h"

#include <random>

namespace Cuda
{
    __host__ __device__ BidirectionalTransform::BidirectionalTransform()
    {
        jitterable.trans = vec3(0.0f);
        jitterable.rot = vec3(0.0f);
        jitterable.scale = vec3(1.0f);
        
        MakeIdentity();
    }

    __host__ void BidirectionalTransform::Randomise(const vec2& range)
    {
        jitterable.trans.Randomise(range);
        jitterable.rot.Randomise(range);
        jitterable.scale.Randomise(range);

        EvaulateJitterables();
    }
    
    __host__ void BidirectionalTransform::FromJson(const ::Json::Node& node, const uint flags)
    {
        const auto transNode = node.GetChildObject("transform", flags);
        if (!transNode) { return; }        

        jitterable.trans.FromJson("pos", transNode, flags);
        jitterable.rot.FromJson("rot", transNode, ::Json::kSilent);
        jitterable.scale.FromJson("sca", transNode, ::Json::kSilent);

        EvaulateJitterables();
        
        // Build the transform
        Create(trans, rot, scale);
    }

    __host__ void BidirectionalTransform::ToJson(Json::Node& parentNode) const
    {
        auto transNode = parentNode.AddChildObject("transform");

        jitterable.trans.ToJson("pos", transNode);
        jitterable.rot.ToJson("rot", transNode);
        jitterable.scale.ToJson("sca", transNode);
    }

    __host__ BidirectionalTransform::BidirectionalTransform(const ::Json::Node& node, const uint flags)
    {
        FromJson(node, flags);
    }

    __host__ void BidirectionalTransform::EvaulateJitterables()
    {
        trans = jitterable.trans.Evaluate();
        rot = jitterable.rot.Evaluate();
        scale = jitterable.scale.Evaluate();
    }
}