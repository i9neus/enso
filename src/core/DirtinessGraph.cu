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
        DirtinessGraph::Get().OnDirty(flag, GetAssetHandle());
    }

    __host__ void Host::Dirtyable::SignalDirty(const std::vector<DirtinessKey>& flagList)
    {
        auto& graph = DirtinessGraph::Get();
        for (const auto& flag : flagList)
        {
            m_dirtyFlags.emplace(flag);
            graph.OnDirty(flag, GetAssetHandle());
        }
    }

    __host__ void Host::Dirtyable::Listen(const std::vector<DirtinessKey>& flagList)
    {
        // Get the asset handle for this object and upcast it
        auto& graph = DirtinessGraph::Get();
        for (const auto& flag : flagList)
        {
            graph.AddListener(GetAssetID(), GetAssetHandle(), flag);
        }
    }

    __host__ bool Host::Dirtyable::IsDirty(const DirtinessKey& id) const
    {
        return m_dirtyFlags.count(id);
    }

    __host__ bool Host::Dirtyable::IsAnyDirty(const std::vector<DirtinessKey>& flagList) const
    {
        for (const auto& flag : flagList)
        {
            if (m_dirtyFlags.count(flag)) return true;
        }
        return false;
    }

    __host__ bool Host::Dirtyable::IsAllDirty(const std::vector<DirtinessKey>& flagList) const
    {
        for (const auto& flag : flagList)
        {
            if (!m_dirtyFlags.count(flag)) return false;
        }
        return true;
    }

    __host__ void Host::Dirtyable::Clean()
    {
        // NOTE: Clearing flags does not trigger a signal to listeners
        OnClean();
        m_dirtyFlags.clear();
    }

    __host__ Host::DirtinessGraph::Listener::Listener(const DirtinessKey& _flag, WeakAssetHandle<Host::Dirtyable>& _handle, EventDeligate& deligate) :
        flag(_flag),
        handle(_handle),
        deligate(deligate)
    {
        // We cache the naked pointer for the hash function
        hash = std::hash<void*>{}((void*)handle.lock().get());
        hash = HashCombine(hash, size_t(flag));
    }

    Host::DirtinessGraph& Host::DirtinessGraph::Get()
    {
        // FIXME: Don't use a singleton. Intead use an instance passed as part of the initialisation context of the scene. 
        static Host::DirtinessGraph singleton;
        return singleton;
    }

    __host__ void Host::DirtinessGraph::OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller)
    {
        //m_eventSet.emplace(flag);
        std::lock_guard<std::mutex> lock(m_mutex);

        int numExpired = 0;
        for (auto it = m_listenerFromFlag.find(flag); it != m_listenerFromFlag.end() && it->first == flag; ++it)
        {
            auto& listener = it->second;
            if (!listener)
            {
                auto expiredIt = it;
                ++it;
                m_listenerFromFlag.erase(expiredIt);
                ++numExpired;
            }
            else
            {                
                if (listener.deligate)
                {
                    // If a custom deligate has been specified, call it here
                    listener.deligate(flag, caller);
                }
                else
                {
                    // Otherwise, just call the default method
                    listener.handle.lock()->OnDirty(flag, caller);
                }
            }
        }

        // If any of the handles expired, do some garbage collection
        if (numExpired > 0)
        {
            Log::Warning("Garbage collection removed %i listeners from dirtiness graph.", numExpired);
        }
    }

    __host__ bool Host::DirtinessGraph::AddListener(const std::string& assetId, WeakAssetHandle<Host::Asset>& assetHandle, const DirtinessKey& flag, EventDeligate functor)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // Check to make sure this object isn't already listening for this event
        for (auto listener = m_flagFromAssetId.find(assetId); listener != m_flagFromAssetId.end() && listener->first == assetId; ++listener)
        {
            if (listener->second == flag)
            {
                Log::Error("Warning: object '%s' is already listening for event %i", assetId, flag);
                return false;
            }
        }

        // Upcast the weak asset handle to a weak dirtyable handle
        Assert(!assetHandle.expired());
        auto dirtyableHandle = AssetHandle<Host::Asset>(assetHandle).DynamicCast<Host::Dirtyable>().GetWeakHandle();

        // Emplace the new listener
        m_listenerFromFlag.emplace(flag, Listener(flag, dirtyableHandle, functor));
        m_flagFromAssetId.emplace(assetId, flag);

        Log::Error("Object '%s' added as a listener to the dirtiness graph.", assetId);
        return true;
    }

}