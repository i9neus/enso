#include "DirtinessGraph.cuh"
#include "Asset.cuh"

namespace Enso
{           
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

        DirtinessGraph::Get().OnDirty(flag, *this);
    }

    __host__ void Host::Dirtyable::Listen(const DirtinessKey& flag)
    {// Get the asset handle for this object and upcast it
        WeakAssetHandle<Host::Dirtyable> weakHandle = std::dynamic_pointer_cast<Host::Dirtyable>(GetAssetHandle().lock());
        DirtinessGraph::Get().AddListener(GetAssetID(), weakHandle, flag);
    }

    __host__ bool Host::Dirtyable::IsDirty(const DirtinessKey& id) const
    {
        return m_dirtyFlags.count(id);
    }

    __host__ void Host::Dirtyable::Clean()
    {
        // NOTE: Clearing flags does not trigger a signal to listeners
        m_dirtyFlags.clear();
    }

    __host__ Host::DirtinessGraph::Listener::Listener(const DirtinessKey& _flag, WeakAssetHandle<Dirtyable>& _handle, EventDeligate& deligate) :
        m_flag(_flag),
        m_handle(_handle),
        m_deligate(deligate)
    {
        // We cache the naked pointer for the hash function
        Assert(!m_handle.expired());
        m_hash = std::hash<void*>{}(m_handle.lock().get());
        m_hash = HashCombine(m_hash, size_t(m_flag));
    }

    __host__ void Host::DirtinessGraph::Listener::OnDirty(Host::Dirtyable& caller)
    {
        Assert(!m_handle.expired());

        if (m_deligate)
        {
            // If a custom deligate has been specified, call it here
            m_deligate(m_flag, caller);
        }
        else
        {
            // Otherwise, just call the default method
            m_handle.lock()->OnDirty(m_flag, caller);
        }
    }

    Host::DirtinessGraph& Host::DirtinessGraph::Get()
    {
        // FIXME: Don't use a singleton. Intead use an instance passed as part of the initialisation context of the scene. 
        static Host::DirtinessGraph singleton;
        return singleton;
    }

    __host__ void Host::DirtinessGraph::OnDirty(const DirtinessKey& flag, Host::Dirtyable& caller)
    {
        //m_eventSet.emplace(flag);
        std::lock_guard<std::mutex> lock(m_mutex);

        int numExpired = 0;
        for (auto listener = m_listenerFromFlag.find(flag); listener != m_listenerFromFlag.end() && listener->first == flag; ++listener)
        {
            if (!listener->second)
            {
                auto expiredIt = listener;
                ++listener;
                m_listenerFromFlag.erase(expiredIt);
                ++numExpired;
            }
            else
            {
                listener->second.OnDirty(caller);
            }
        }

        // If any of the handles expired, do some garbage collection
        if (numExpired > 0)
        {
            Log::Warning("Garbage collection removed %i listeners from dirtiness graph.", numExpired);
        }
    }

    __host__ bool Host::DirtinessGraph::AddListener(const std::string& assetId, WeakAssetHandle<Host::Dirtyable>& handle, const DirtinessKey& flag, EventDeligate functor)
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

        m_listenerFromFlag.emplace(flag, Listener(flag, handle, functor));
        m_flagFromAssetId.emplace(assetId, flag);

        Log::Error("Object '%s' added as a listener to the dirtiness graph.", assetId);
        return true;
    }

}