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

    __host__ void Host::Dirtyable::SetDirty(const DirtinessKey& flag)
    {
        m_dirtyFlags.emplace(flag);
    }

    __host__ void Host::Dirtyable::SignalDirty(const DirtinessKey& flag)
    {
        m_dirtyFlags.emplace(flag);
        s_dirtinessGraph.OnDirty(flag);
    }

    __host__ void Host::Dirtyable::Listen(const DirtinessKey& flag)
    {
        // Get the asset handle for this object and upcast it
        WeakAssetHandle<Host::Dirtyable> weakHandle = std::dynamic_pointer_cast<Host::Dirtyable>(GetAssetHandle().lock());
        s_dirtinessGraph.AddListener(GetAssetID(), weakHandle, flag);
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
        for (const auto& event : m_eventSet)
        {
            for (auto listener = m_handleFromFlag.find(event); listener != m_handleFromFlag.end() && listener->first == event; ++listener)
            {
                if (listener->second.expired())
                {
                    auto expiredIt = listener;
                    ++listener;
                    m_handleFromFlag.erase(expiredIt);
                    ++numExpired;
                }
                else
                {
                    listener->second.lock()->OnDirty(event);
                }
            }
        }

        // Empty the queue
        std::lock_guard<std::mutex> lock(m_mutex);
        m_eventSet.clear();

        // If any of the handles expired, do some garbage collection
        if (numExpired > 0)
        {
            Log::Warning("Garbage collection removed %i listeners from dirtiness graph.", numExpired);
        }
    }

    __host__ void Host::DirtinessGraph::OnDirty(const DirtinessKey& flag)
    {
        m_eventSet.emplace(flag);
    }

    __host__ bool Host::DirtinessGraph::AddListener(const std::string& assetId, WeakAssetHandle<Host::Dirtyable>& handle, const DirtinessKey& flag)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        AssertMsgFmt(!handle.expired(), "Handle for asset '%s' has expired.", assetId.c_str());

        // Check to make sure this object isn't already listening for this event
        for (auto listener = m_flagFromAssetId.find(assetId); listener != m_flagFromAssetId.end() && listener->first == assetId; ++listener)
        {
            if (listener->second == flag)
            {
                Log::Error("Warning: object '%s' is already listening for event %i", assetId, flag);
                return false;
            }
        }

        m_handleFromFlag.emplace(flag, handle);
        m_flagFromAssetId.emplace(assetId, flag);

        Log::Error("Object '%s' added as a listener to the dirtiness graph.", assetId);
        return true;
    }

}