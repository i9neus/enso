#include "DirtinessGraph.cuh"

namespace Enso
{           
    std::recursive_mutex Host::Dirtyable::m_dirtyMutex;
    
    __host__ Host::Dirtyable::Dirtyable(const Asset::InitCtx& initCtx) :
        Asset(initCtx)
    {

    }

    __host__ Host::Dirtyable::~Dirtyable()
    {
        // Clean up by removing any listeners
        DirtinessGraph::Get().RemoveAllListeners(GetAssetID());
    }

    __host__ void Host::Dirtyable::SetDirty(const DirtinessEvent& event)
    {
        std::lock_guard<std::recursive_mutex> lock(m_dirtyMutex);
        m_dirtyEvents.insert(event);
    }

    __host__ void Host::Dirtyable::SetDirty(const std::vector<DirtinessEvent>& eventList)
    {
        std::lock_guard<std::recursive_mutex> lock(m_dirtyMutex);
        m_dirtyEvents.insert(eventList.begin(), eventList.end());
    }

    __host__ void Host::Dirtyable::UnsetDirty(const DirtinessEvent& event)
    {
        std::lock_guard<std::recursive_mutex> lock(m_dirtyMutex);

        m_dirtyEvents.erase(event);
    }

    __host__ void Host::Dirtyable::UnsetDirty(const std::vector<DirtinessEvent>& eventList)
    {
        std::lock_guard<std::recursive_mutex> lock(m_dirtyMutex);
        for (auto event : eventList)
        {
            m_dirtyEvents.erase(event);
        }
    }

    /*__host__ void Host::Dirtyable::SignalDirty()
    {
        auto& graph = DirtinessGraph::Get();
        for (const auto& event : m_dirtyEvents)
        {
            graph.OnDirty(event, GetAssetHandle());
        }
    }*/

    __host__ void Host::Dirtyable::SignalDirty(const DirtinessEvent& event)
    {
        AssetHandle<Host::Asset> handle(GetAssetHandle());
        AssertMsgFmt(handle, "GetAssetHandle() returned expired pointer for asset '%s'", GetAssetID());

        if (m_dirtyEvents.find(event) == m_dirtyEvents.end())
        {
            m_dirtyEvents.emplace(event);
            DirtinessGraph::Get().OnDirty(event, handle);
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
            graph.AddListener(GetAssetID(), GetAssetHandle(), event, kListenerNotify);
        }
    }

    __host__ void Host::Dirtyable::Cascade(const std::vector<DirtinessEvent>& eventList)
    {
        // Get the asset handle for this object and upcast it
        auto& graph = DirtinessGraph::Get();
        for (const auto& event : eventList)
        {
            graph.AddListener(GetAssetID(), GetAssetHandle(), event, kListenerCascade);
        }
    }

    __host__ bool Host::Dirtyable::IsDirty(const DirtinessEvent& id) const
    {
        std::lock_guard<std::recursive_mutex> lock(m_dirtyMutex);
        return m_dirtyEvents.count(id);
    }

    __host__ bool Host::Dirtyable::IsAnyDirty(const std::vector<DirtinessEvent>& eventList) const
    {
        std::lock_guard<std::recursive_mutex> lock(m_dirtyMutex);
        for (const auto& event : eventList)
        {
            if (m_dirtyEvents.count(event)) return true;
        }
        return false;
    }

    __host__ bool Host::Dirtyable::IsAllDirty(const std::vector<DirtinessEvent>& eventList) const
    {
        std::lock_guard<std::recursive_mutex> lock(m_dirtyMutex);
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

        std::lock_guard<std::recursive_mutex> lock(m_dirtyMutex);
        m_dirtyEvents.clear();
    }

    __host__ Host::DirtinessGraph::Listener::Listener(const DirtinessEvent& _event, WeakAssetHandle<Host::Dirtyable>& _handle, const int _callbackType, EventDeligate& deligate) :
        event(_event),
        handle(_handle),
        deligate(deligate),
        callbackType(_callbackType)
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

    __host__ void Host::DirtinessGraph::OnDirty(const DirtinessEvent& event, AssetHandle<Host::Asset>& caller)
    {
        //m_eventSet.emplace(event);

        int numExpired = 0;
        for (auto it = m_listenerFromEvent.find(event); it != m_listenerFromEvent.end() && it->first == event; ++it)
        {
            auto& listener = it->second;
            AssetHandle<Host::Dirtyable> handle(listener.handle);

            if (!handle)
            {
                auto expiredIt = it;
                ++it;
                m_listenerFromEvent.erase(expiredIt);
                ++numExpired;
            }
            else
            {
                switch (listener.callbackType)
                {
                case kListenerCascade:
                    handle->SetDirty(event); 
                    break;

                case kListenerNotify:
                    handle->OnDirty(event, caller); 
                    break;

                case kListenerDelegate:
                    Assert(listener.deligate);
                    listener.deligate(event, caller);
                    break;

                default:
                    Assert(false);
                }
            }
        }

        // If any of the handles expired, do some garbage collection
        if (numExpired > 0)
        {
            Log::Error("Warning: Garbage collection removed %i listeners from dirtiness graph. This shouldn't happen!", numExpired);
        }
    }

    __host__ void Host::DirtinessGraph::RemoveAllListeners(const std::string& assetId)
    {
        if (m_eventFromAssetId.erase(assetId) > 0)
        {
            Log::Debug("Removed %i listeners for expired asset %s", assetId);
        }
    }

    __host__ bool Host::DirtinessGraph::AddListener(const std::string& assetId, WeakAssetHandle<Host::Asset>& assetHandle, const DirtinessEvent& event, const int callbackType, EventDeligate functor)
    {                
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

        Assert(uint(m_semaphore) == kGraphUnlocked);
        m_semaphore.TryUntil(kGraphUnlocked, kGraphLocked);
        
        // Emplace the new listener
        m_listenerFromEvent.emplace(event, Listener(event, dirtyableHandle, callbackType, functor));
        m_eventFromAssetId.emplace(assetId, event);

        m_semaphore.TryUntil(kGraphLocked, kGraphUnlocked);

        Log::Error("Object '%s' added as a listener to the dirtiness graph.", assetId);
        return true;
    }

}