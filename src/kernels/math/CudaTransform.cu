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
    }
    
    __host__ void BidirectionalTransform::FromJson(const ::Json::Node& node, const uint flags)
    {
        const auto transNode = node.GetChildObject("transform", flags);
        if (!transNode) { return; }        

        trans.FromJson("pos", transNode, flags);
        rot.FromJson("rot", transNode, ::Json::kSilent);
        scale.FromJson("sca", transNode, ::Json::kSilent);

        //scale.eval.y = scale.eval.z = scale.eval.x;
        
        // Build the transform
        Create(trans(), rot(), scale());
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
}