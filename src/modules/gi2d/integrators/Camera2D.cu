#include "Camera2D.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __host__ bool Host::Camera2D::Serialise(Json::Node& node, const int flags) const
    {
        Json::Node tracableNode = node.AddChildObject("camera");
        SceneObject::Serialise(tracableNode, flags);

        return true;
    }

    __host__ uint Host::Camera2D::Deserialise(const Json::Node& node, const int flags)
    {
        Json::Node tracableNode = node.GetChildObject("camera", flags);
        if (tracableNode) { SceneObject::Deserialise(tracableNode, flags); }

        return m_dirtyFlags;
    }
}