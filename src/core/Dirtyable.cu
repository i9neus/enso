#include "Dirtyable.cuh"
#include "Asset.cuh"

namespace Enso
{       
    // FIXME: Don't use globl state. Intead use an instance passed as part of the initialisation context of the scene. 
    static Host::DirtinessGraph s_dirtinessGraph;
    
    __host__ Host::Dirtyable::Dirtyable(const Asset::InitCtx& initCtx) :
        Asset(initCtx)
    {
    }
    
    __host__ void Host::Dirtyable::Listen(const DirtinessKey& eventId)
    {
        // Get the asset handle for this object and upcast it
        WeakAssetHandle<Host::Dirtyable> weakHandle = std::dynamic_pointer_cast<Host::Dirtyable>(GetAssetHandle().lock());

        s_dirtinessGraph.AddListener(GetAssetID(), weakHandle, eventId);
    }

    __host__ bool Host::Dirtyable::IsDirty(const DirtinessKey& id)
    {
        return m_dirtyFlags.count(id);
    }

    __host__ void Host::Dirtyable::Clean()
    {
        // NOTE: Clearing flags does not trigger a signal to listeners
        m_dirtyFlags.clear();
    }

    __host__ void Host::DirtinessGraph::Flush()
    {
        // Flush the event queue by iterating through all outstanding events and invoking the triggers corresponding to them
        int numExpired = 0;
        for (const auto& event : m_eventQueue)
        {
            for (auto listener = m_listenerHandles.find(event.first); listener != m_listenerHandles.end() && listener->first == event.first; ++listener)
            {
                if (listener->second.expired())
                {
                    auto expiredIt = listener;
                    ++listener;
                    m_listenerHandles.erase(expiredIt);
                    ++numExpired;
                }
                else
                {
                    listener->second.lock()->OnDirty(event.first, listener->second);
                }
            }
        }

        // Empty the queue
        std::lock_guard<std::mutex> lock(m_mutex);
        m_eventQueue.clear();

        // If any of the handles expired, do some garbage collection
        if (numExpired > 0)
        {
            Log::Warning("Garbage collection removed %i listeners from dirtiness graph.", numExpired);
        }
    }

    __host__ bool Host::DirtinessGraph::AddListener(const std::string& id, WeakAssetHandle<Host::Dirtyable>& handle, const DirtinessKey& eventId)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        AssertMsgFmt(!handle.expired(), "Handle for asset '%s' has expired.", id.c_str());

        // Check to make sure this object isn't already listening for this event
        for (auto listener = m_listenerKeys.find(id); listener != m_listenerKeys.end() && listener->first == id; ++listener)
        {
            if (listener->second == eventId)
            {
                Log::Error("Warning: object '%s' is already listening for event %i", id, eventId);
                return false;
            }
        }

        m_listenerHandles.emplace(eventId, handle);
        m_listenerKeys.emplace(id, eventId);

        Log::Debug("Object '%s' added as a listener to the dirtiness graph.");
        return true;
    }

}