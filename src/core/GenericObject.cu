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
    __host__ Host::GenericObject::GenericObject(const std::string& id) :
        Asset(id),
        m_allocator(*this),
        m_renderObjectFlags(0),
        m_dirtyFlags(0),
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
    
    __host__ void Host::GenericObject::RegisterEvent(const std::string& eventID)
    {
        AssertMsgFmt(m_eventRegistry.find(eventID) == m_eventRegistry.end(), "Event '%s' already registered", eventID.c_str());

        m_eventRegistry.emplace(eventID);
    }

    __host__ void Host::GenericObject::Unlisten(const GenericObject& owner, const std::string& eventID)
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

    __host__ void Host::GenericObject::OnEvent(const std::string& eventID)
    {
        for (auto it = m_actionDeligates.find(eventID); it != m_actionDeligates.end() && it->first == eventID; ++it)
        {
            it->second.m_functor(*this, eventID);
        }
    }
}