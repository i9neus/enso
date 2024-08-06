#include "Tracable.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    /*__host__ __device__ bool Device::Tracable::IntersectBBox(const BBox2f& bBox) const
    {
        return bBox.Intersects(m_params.objectBBox);
    }*/

    __host__ Host::Tracable::Tracable(const Asset::InitCtx& initCtx) :
        GenericObject(initCtx)
    {
    }


    __host__ void Host::Tracable::Synchronise(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::Tracable>(cu_deviceInstance, m_params);
        }

        OnSynchroniseTracable(syncFlags);
    }

    __host__ bool Host::Tracable::Serialise(Json::Node& node, const int flags) const
    {
        //Json::Node tracableNode = node.AddChildObject("tracable");

        return true;
    }

    __host__ bool Host::Tracable::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = false;
        //Json::Node tracableNode = node.GetChildObject("tracable", flags);

        return isDirty;
    }
}