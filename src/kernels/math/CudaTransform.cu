#include "CudaTransform.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ void BidirectionalTransform::FromJson(const Json::Node& parentNode)
    {
        const auto transNode = parentNode.GetChildObject("transform", false);
        if (!transNode) { return; }

        if (!transNode.GetVector("trans", trans, false))
        {
            trans = 0.0f;
        }
        if (!transNode.GetVector("rot", rot, false))
        {
            rot = 0.0f;
        }
        if (transNode.GetVector("scale", scale, false))
        {
            scale = 1.0f;
        }
        
        // Build the transform
        Create(trans, rot, scale);
    }

    __host__ void BidirectionalTransform::ToJson(Json::Node& parentNode) const
    {
        auto transNode = parentNode.AddChildObject("transform");

        transNode.AddArray("trans", std::vector<float>({ trans.x, trans.y, trans.z }));
        transNode.AddArray("rot", std::vector<float>({ rot.x, rot.y, rot.z }));
        transNode.AddArray("scale", std::vector<float>({ scale.x, scale.y, scale.z }));
    }

    __host__ BidirectionalTransform::BidirectionalTransform(const Json::Node& node)
    {
        FromJson(node);
    }
}