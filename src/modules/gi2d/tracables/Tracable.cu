#include "Tracable.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __host__ __device__ bool Device::Tracable::IntersectBBox(const BBox2f& bBox) const
    {
        return bBox.Intersects(m_objectBBox);
    }

    __host__ bool Host::Tracable::Serialise(Json::Node& node, const int flags) const
    {
        Json::Node tracableNode = node.AddChildObject("tracable");
        SceneObject::Serialise(tracableNode, flags);

        return true;
    }

    __host__ bool Host::Tracable::Deserialise(const Json::Node& node, const int flags)
    {
        Json::Node tracableNode = node.GetChildObject("tracable", flags);
        if (tracableNode) { SceneObject::Deserialise(tracableNode, flags); }

        return true;
    }
}