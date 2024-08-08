#include "DirtinessGraph.cuh"
#include "Asset.cuh"

namespace Enso
{           
    __host__ Host::Dirtyable::Dirtyable(const Asset::InitCtx& initCtx) :
        Asset(initCtx)
    {
    }

    __host__ void Host::Dirtyable::SetDirty(const DirtinessEvent& event)
    {
        m_dirtyEvents.emplace(event);
    }

    __host__ void Host::Dirtyable::SetDirty(const std::vector<DirtinessEvent>& eventList)
    {
        m_dirtyEvents.insert(eventList.begin(), eventList.end());
    }

    __host__ void Host::Dirtyable::SignalDirty()
    {
        auto& graph = DirtinessGraph::Get();
        for (const auto& event : m_dirtyEvents)
        {
            graph.OnDirty(event, GetAssetHandle());
        }
    }

    __host__ void Host::Dirtyable::SignalDirty(const DirtinessEvent& event)
    {
        if (m_dirtyEvents.find(event) == m_dirtyEvents.end())
        {
            m_dirtyEvents.emplace(event);
            DirtinessGraph::Get().OnDirty(event, GetAssetHandle());
        }
    }

    __host__ void Host::Dirtyable::SignalDirty(const std::vector<DirtinessEvent>& eventList)
    {
        auto& graph = DirtinessGraph::Get();
        for (const auto& event : eventList)
        {
            SignalDirty(event);
        }
    }

    __host__ void Host::Dirtyable::Listen(const std::vector<DirtinessEvent>& eventList)
    {
        // Get the asset handle for this object and upcast it
        auto& graph = DirtinessGraph::Get();
        for (const auto& event : eventList)
        {
            graph.AddListener(GetAssetID(), GetAssetHandle(), event);
        }
    }

    __host__ bool Host::Dirtyable::IsDirty(const DirtinessEvent& id) const
    {
        return m_dirtyEvents.count(id);
    }

    __host__ bool Host::Dirtyable::IsAnyDirty(const std::vector<DirtinessEvent>& eventList) const
    {
        for (const auto& event : eventList)
        {
            if (m_dirtyEvents.count(event)) return true;
        }
        return false;
    }

    __host__ bool Host::Dirtyable::IsAllDirty(const std::vector<DirtinessEvent>& eventList) const
    {
        for (const auto& event : eventList)
        {
            if (!m_dirtyEvents.count(event)) return false;
        }
        return true;
    }

    __host__ void Host::Dirtyable::Clean()
    {
        // NOTE: Clearing events does not trigger a signal to listeners
        OnClean();
        m_dirtyEvents.clear();
    }

    __host__ Host::DirtinessGraph::Listener::Listener(const DirtinessEvent& _event, WeakAssetHandle<Host::Dirtyable>& _handle, EventDeligate& deligate) :
        event(_event),
        handle(_handle),
        deligate(deligate)
    {
        // We cache the naked pointer for the hash function
        hash = std::hash<void*>{}((void*)handle.lock().get());
        hash = HashCombine(hash, size_t(event));
    }

    Host::DirtinessGraph& Host::DirtinessGraph::Get()
    {
        // FIXME: Don't use a singleton. Intead use an instance passed as part of the initialisation context of the scene. 
        static Host::DirtinessGraph singleton;
        return singleton;
    }

    __host__ Host::DirtinessEvent Host::DirtinessGraph::RegisterEvent(const std::string& id, const bool mustExist)
    {
        Host::DirtinessGraph& graph = Get();
        DirtinessEvent eventHash = std::hash<std::string>{}(id); 
        const bool exists = graph.m_eventHashFromId.find(id) == graph.m_eventHashFromId.end();
        
        AssertMsgFmt(!mustExist || exists, "Event '%s' was not registered prior to its handle being requested", id);       

        if (!exists) 
        {
            Log::Debug("Registered dirtiness event '%s' with hash 0x%x", id, eventHash);
            graph.m_eventHashFromId[id] = eventHash;
        }

        return eventHash;
    }

    __host__ void Host::DirtinessGraph::OnDirty(const DirtinessEvent& event, WeakAssetHandle<Host::Asset>& caller)
    {
        //m_eventSet.emplace(event);
        std::lock_guard<std::mutex> lock(m_mutex);

        int numExpired = 0;
        for (auto it = m_listenerFromEvent.find(event); it != m_listenerFromEvent.end() && it->first == event; ++it)
        {
            auto& listener = it->second;
            if (!listener)
            {
                auto expiredIt = it;
                ++it;
                m_listenerFromEvent.erase(expiredIt);
                ++numExpired;
            }
            else
            {
                if (listener.deligate)
                {
                    // If a custom deligate has been specified, call it here
                    listener.deligate(event, caller);
                }
                else
                {
                    // Otherwise, just call the default method
                    listener.handle.lock()->OnDirty(event, caller);
                }
            }
        }

        // If any of the handles expired, do some garbage collection
        if (numExpired > 0)
        {
            Log::Warning("Garbage collection removed %i listeners from dirtiness graph.", numExpired);
        }
    }

    __host__ bool Host::DirtinessGraph::AddListener(const std::string& assetId, WeakAssetHandle<Host::Asset>& assetHandle, const DirtinessEvent& event, EventDeligate functor)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // Check to make sure this object isn't already listening for this event
        for (auto listener = m_eventFromAssetId.find(assetId); listener != m_eventFromAssetId.end() && listener->first == assetId; ++listener)
        {
            if (listener->second == event)
            {
                Log::Error("Warning: object '%s' is already listening for event %i", assetId, event);
                return false;
            }
        }

        // Upcast the weak asset handle to a weak dirtyable handle
        Assert(!assetHandle.expired());
        auto dirtyableHandle = AssetHandle<Host::Asset>(assetHandle).DynamicCast<Host::Dirtyable>().GetWeakHandle();

        // Emplace the new listener
        m_listenerFromEvent.emplace(event, Listener(event, dirtyableHandle, functor));
        m_eventFromAssetId.emplace(assetId, event);

        Log::Error("Object '%s' added as a listener to the dirtiness graph.", assetId);
        return true;
    }

}