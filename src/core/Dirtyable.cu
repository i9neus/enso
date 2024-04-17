#include "Dirtyable.cuh"

namespace Enso
{
    __host__ void Dirtyable::SetDirty(const DirtinessKey& id)
    {
        if (m_dirtyFlags.count(id) == 0)
        {
            m_dirtyFlags.emplace(id);
            m_dirtinessGraph.OnEvent(id);
        }
    }

    __host__ bool Dirtyable::IsDirty(const DirtinessKey& id)
    {
        return m_dirtyFlags.count(id);
    }

    __host__ void Dirtyable::Clean()
    {
        // NOTE: Clearing flags does not trigger a signal to listeners
        m_dirtyFlags.clear();
    }

    /*__host__ bool DirtinessGraph::IsDirty(const DirtinessKey& flag) const
    {
        return m_dirtyFlags.count(flag);
    }*/

    __host__ void DirtinessGraph::OnEvent(const DirtinessKey& flag)
    {
        
    }

    __host__ void DirtinessGraph::Flush()
    {
        // 
        for (const auto& event : m_eventQueue)
        {
            for (auto it = m_listeners.find(event); it != m_listeners.end() && it->first == event; ++it)
            {
                
                Listener& listener = it->second;
                listener.deligate(listener.owner, flag);
            }
        }
    }

    __host__ void AddListener(Dirtyable& owner, const DirtinessKey& eventID)
    {
        if (!ListenerExists(owner, eventID))
        {
            m_listeners.emplace(eventID, Listener(owner, DirtinessDeligate(nullptr)));
        }
    }

    __host__ bool DirtinessGraph::ListenerExists(Dirtyable& owner, const DirtinessKey& eventID) const
    {
        // Check whether this object is already listening for this event
        for (auto it = m_listeners.find(eventID); it != m_listeners.end() && it->first == eventID; ++it)
        {
            if (&it->second == static_cast<Dirtyable*>(&owner))
            {
                //Log::Error("Internal error: deligate '%s' ['%s' -> '%s'] is already registered", eventID, GetAssetID(), owner.GetAssetID());
                return true;
            }
        }
        return false;
    }

}