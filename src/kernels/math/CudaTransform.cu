#include "CudaTransform.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ void BidirectionalTransform::FromJson(const ::Json::Node& node, const uint flags)
    {
        const auto transNode = node.GetChildObject("transform", flags);
        if (!transNode) { return; }

        trans = rot = 0.0f;
        scale = 1.0f;

        transNode.GetVector("pos", trans, flags);       
        transNode.GetVector("rot", rot, flags);
        transNode.GetVector("sca", scale, flags);

        // Convert from degrees to radians
        rot = toRad(rot);
        
        // Build the transform
        Create(trans, rot, scale);
    }

    __host__ void BidirectionalTransform::ToJson(Json::Node& parentNode) const
    {
        auto transNode = parentNode.AddChildObject("transform");

        transNode.AddArray("pos", std::vector<float>({ trans.x, trans.y, trans.z }));
        transNode.AddArray("rot", std::vector<float>({ toDeg(rot.x), toDeg(rot.y), toDeg(rot.z) }));
        transNode.AddArray("sca", std::vector<float>({ scale.x, scale.y, scale.z }));
    }

    __host__ BidirectionalTransform::BidirectionalTransform(const ::Json::Node& node, const uint flags)
    {
        FromJson(node, flags);
    }

    __host__ bool BidirectionalTransform::operator==(const BidirectionalTransform& rhs) const
    {
        return trans == rhs.trans && rot == rhs.rot && scale == rhs.scale;
    }
}