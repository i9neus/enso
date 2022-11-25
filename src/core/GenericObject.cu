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

    __host__ uint RenderObjectParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        return this->flags.FromJson("objectFlags", node, flags);
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
    
    __host__ void Host::RenderObject::RegisterEvent(const std::string& eventID)
    {
        AssertMsgFmt(m_eventRegistry.find(eventID) == m_eventRegistry.end(), "Event '%s' already registered", eventID.c_str());

        m_eventRegistry.emplace(eventID);
    }

    __host__ void Host::RenderObject::Unlisten(const RenderObject& owner, const std::string& eventID)
    {
        for (auto it = m_actionDeligates.find(eventID); it != m_actionDeligates.end() && it->first == eventID; ++it)
        {
            if (&it->second.m_owner == &owner)
            {
                m_actionDeligates.erase(it);
                return;
            }
            ++it;
        }

        Log::Error("Internal error: deligate '%s' ['%s' -> '%s'] can't be deregistered because it does not exist.", eventID, GetAssetID(), owner.GetAssetID());
    }

    __host__ void Host::RenderObject::OnEvent(const std::string& eventID)
    {
        for (auto it = m_actionDeligates.find(eventID); it != m_actionDeligates.end() && it->first == eventID; ++it)
        {
            it->second.m_functor(*this, eventID);
        }
    }
}