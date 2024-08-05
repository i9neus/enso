#pragma once

#include "math/Math.cuh"
#include "io/Serialisable.cuh"
#include "AssetAllocator.cuh"
#include "math/Hash.cuh"
#include "DirtinessFlags.cuh"

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
            using EventDeligate = std::function<void(const DirtinessKey& id, WeakAssetHandle<Host::Asset>& caller)>;

        private:           
            class Listener
            {
            private:
                size_t hash;

            public:
                WeakAssetHandle<Host::Dirtyable>  handle;
                EventDeligate               deligate;
                const DirtinessKey          flag;

            public:
                __host__ Listener(const DirtinessKey& _flag, WeakAssetHandle<Host::Dirtyable>& _handle, EventDeligate& _functor);

                __host__ operator bool() const { return !handle.expired(); }
                __host__ bool operator <(const Listener& rhs) const { return hash < rhs.hash; }
            };
            
            std::mutex                                                  m_mutex;
            std::multimap <DirtinessKey, Listener>                      m_listenerFromFlag;
            std::multimap <std::string, DirtinessKey>                   m_flagFromAssetId;

        public:

            __host__ static DirtinessGraph& Get();
            
            __host__ bool AddListener(const std::string& id, WeakAssetHandle<Host::Asset>& handle, const DirtinessKey& flag, EventDeligate functor = nullptr);
            __host__ void OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller);

        private:
            __host__ DirtinessGraph() = default;
        };

        class Dirtyable : public Host::Asset
        {
            friend DirtinessGraph;
        public:
            __host__ Dirtyable(const Asset::InitCtx& initCtx);

            __host__ bool               IsClean() const { return m_dirtyFlags.empty(); }
            __host__ bool               IsDirty(const DirtinessKey& flag) const;
            __host__ bool               IsAnyDirty(const std::vector<DirtinessKey>& flagList) const;
            __host__ bool               IsAllDirty(const std::vector<DirtinessKey>& flagList) const;
            __host__ void               Clean();

        protected:
            __host__ virtual void       OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller) {}
            __host__ virtual void       OnClean() {}

            __host__ void               SetDirty(const DirtinessKey& flag);
            __host__ void               SetDirty(const std::vector<DirtinessKey>& flagList);

            __host__ void               SignalDirty();
            __host__ void               SignalDirty(const DirtinessKey& flag);
            __host__ void               SignalDirty(const std::vector<DirtinessKey>& flagList);

            //__host__ void               Listen(const DirtinessKey& flag) { Listen({ flag }); }
            __host__ void               Listen(const std::vector<DirtinessKey>& flagList);

            template<typename SuperClass, typename Deligate>
            __host__ __forceinline__ void Listen(const DirtinessKey& flag, SuperClass& super, Deligate deligate)
            {
                DirtinessGraph::EventDeligate functor(std::bind(deligate, &super, std::placeholders::_1, std::placeholders::_2));
                DirtinessGraph::Get().AddListener(GetAssetID(), GetAssetHandle(), flag, functor);
            }

        private:
            std::set<DirtinessKey>      m_dirtyFlags;
        }; 
    }
}

namespace std
{
    template<typename ObjectType>
    std::size_t operator==(const Enso::AssetHandle<ObjectType>& lhs, const Enso::AssetHandle<ObjectType>& rhs)
    {
        const std::size_t lHash = (lhs.expired()) ? std::size_t(0) : std::hash<ObjectType*>{}(lhs.lock().get());
        const std::size_t rHash = (rhs.expired()) ? std::size_t(0) : std::hash<ObjectType*>{}(rhs.lock().get());
        return lHash == rHash;
    }
}

template<typename ObjectType>
struct std::hash<Enso::AssetHandle<ObjectType>>
{
    // Injected specialisation so we can has weak asset pointers
    std::size_t operator()(const Enso::AssetHandle<ObjectType>& handle) const noexcept
    {
        return (handle.expired()) ? std::size_t(0) : std::hash<ObjectType*>{}(handle.lock().get());
    }
};