#include "CudaRenderObject.cuh"
#include "generic/JsonUtils.h"
#include "generic/FilesystemUtils.h"

namespace Cuda
{
    __host__ __device__ RenderObjectParams::RenderObjectParams() :
        flags(0, 2) {}

    __host__ void RenderObjectParams::ToJson(::Json::Node& node) const
    {
        flags.ToJson("objectFlags", node);
    }

    __host__ void RenderObjectParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        this->flags.FromJson("objectFlags", node, flags);
    }

    __host__ void RenderObjectParams::Randomise(const vec2& range)
    {
        flags.Update(kJitterRandomise);
    }
    
    __host__ void Host::RenderObject::UpdateDAGPath(const ::Json::Node& node)
    {
        if (!node.HasDAGPath())
        {
            Log::Error("Internal error: JSON node for '%s' has no DAG path.\n", GetAssetID());
            return;
        }

        SetDAGPath(node.GetDAGPath());
    }
}