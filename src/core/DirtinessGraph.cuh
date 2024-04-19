#pragma once

#include "math/Math.cuh"
#include "io/Serialisable.cuh"
#include "AssetAllocator.cuh"
#include "Hash.h"

#include <set>
#include <map>
#include <functional>
#include <mutex>

namespace Enso
{
    namespace Host
    {       
        class GenericObject;
        class DirtinessGraph;
        class Dirtyable;

        using DirtinessKey = uint;     

        class DirtinessGraph
        {
        public:
            using EventDeligate = std::function<void(const DirtinessKey& id, Host::Dirtyable& caller)>;

        private:           
            class Listener
            {
            private:
                size_t m_hash;

            public:
                WeakAssetHandle<Dirtyable>  m_handle;
                EventDeligate               m_deligate;
                const DirtinessKey          m_flag;

            public:
                __host__ Listener(const DirtinessKey& _flag, WeakAssetHandle<Dirtyable>& _handle, EventDeligate& _functor);

                __host__ void OnDirty(Host::Dirtyable& caller);

                __host__ operator bool() const { return !m_handle.expired(); }
                __host__ bool operator <(const Listener& rhs) const { return m_hash < rhs.m_hash; }
            };
            
            std::mutex                                                  m_mutex;
            std::multimap <DirtinessKey, Listener>                      m_listenerFromFlag;
            std::multimap <std::string, DirtinessKey>                   m_flagFromAssetId;

        public:

            __host__ static DirtinessGraph& Get();
            
            __host__ bool AddListener(const std::string& id, WeakAssetHandle<Host::Dirtyable>& handle, const DirtinessKey& flag, EventDeligate functor = nullptr);
            __host__ void OnDirty(const DirtinessKey& flag, Host::Dirtyable& caller);

        private:
            __host__ DirtinessGraph() = default;
        };

        class Dirtyable : public Host::Asset
        {
            friend DirtinessGraph;
        public:
            __host__ Dirtyable(const Asset::InitCtx& initCtx);

            __host__ bool               IsDirty(const DirtinessKey& flag) const;
            __host__ bool               IsClean() const { return m_dirtyFlags.empty(); }
            __host__ void               Clean();

        protected:
            __host__ virtual void       OnDirty(const DirtinessKey& flag, Host::Dirtyable& caller) {}
            __host__ void               SetDirty(const DirtinessKey& flag);
            __host__ void               SignalDirty(const DirtinessKey& flag);

            __host__ void               Listen(const DirtinessKey& flag);

            template<typename SuperClass, typename Deligate>
            __host__ __forceinline__ void Listen(const DirtinessKey& flag, SuperClass& super, Deligate deligate)
            {
                DirtinessGraph::EventDeligate functor(std::bind(deligate, &super, std::placeholders::_1, std::placeholders::_2));

                // Get the asset handle for this object and upcast it
                WeakAssetHandle<Host::Dirtyable> weakHandle = std::dynamic_pointer_cast<Host::Dirtyable>(GetAssetHandle().lock());
                DirtinessGraph::Get().AddListener(GetAssetID(), weakHandle, flag, functor);
            }

        private:
            std::set<DirtinessKey>      m_dirtyFlags;
        };

    }
}