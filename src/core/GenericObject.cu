#include "GenericObject.cuh"
#include "io/json/JsonUtils.h"
#include "io/FilesystemUtils.h"

namespace Enso
{    
    __host__ __device__ GenericObjectParams::GenericObjectParams()
        {}

    __host__ void GenericObjectParams::ToJson(Json::Node& node) const
    {
        //flags.ToJson("objectFlags", node);
    }

    __host__ uint GenericObjectParams::FromJson(const Json::Node& node, const uint flags)
    {
        return 0u;
    }

    __host__ void GenericObjectParams::Randomise(const vec2& range)
    {
    }

    __host__ Host::GenericObject::GenericObject(const Asset::InitCtx& initCtx) :
        Dirtyable(initCtx),
        m_genericObjectFlags(0),
        m_isFinalised(false),
        m_isConstructed(false)
    {
    }
    
    __host__ void Host::GenericObject::UpdateDAGPath(const Json::Node& node)
    {
        if (!node.HasDAGPath())
        {
            Log::Error("Internal error: JSON node for '%s' has no DAG path.\n", GetAssetID());
            return;
        }

        SetDAGPath(node.GetDAGPath());
    }
}