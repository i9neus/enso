#pragma once

#include "core/math/Math.cuh"
#include "io/Serialisable.cuh"
#include "core/assets/AssetAllocator.cuh"
#include "core/math/Hash.cuh"
#include "DirtinessFlags.cuh"
#include "core/utils/Semaphore.h"

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

        /*class DirtinessEvent
        {
        public:
            DirtinessEvent(const std::string& id) : m_hash(std::hash<std::string>{}(id)) {}

            __host__ inline operator size_t() const { return m_hash; }
            __host__ bool operator==(const DirtinessEvent& rhs) const { return m_hash == rhs.m_hash; }
            __host__ bool operator!=(const DirtinessEvent& rhs) const { return m_hash != rhs.m_hash; }
            __host__ bool operator <(const DirtinessEvent& rhs) const { return m_hash < rhs.m_hash; }

        private:
            size_t m_hash;
        };*/

        using DirtinessEvent = size_t;

        enum ListenerCallbackType : int
        {
            kListenerCascade,   // Cascading means that the listening object simply inherits the dirtiness flag and not directly notified about it
            kListenerNotify,    // Notification calls the default OnDirty() method of the Dirtyable object
            kListenerDelegate   // Delegation means that the listening object specifies a customer functor that's called 
        };

        class DirtinessGraph
        {
        public:
            using EventDeligate = std::function<void(const DirtinessEvent& id, AssetHandle<Host::Asset>& caller)>;

        private:
            class Listener
            {
            private:
                size_t hash;

            public:
                WeakAssetHandle<Host::Dirtyable>    handle;
                EventDeligate                       deligate;
                const DirtinessEvent                event;
                const int                           callbackType;

            public:
                __host__ Listener(const DirtinessEvent& _event, WeakAssetHandle<Host::Dirtyable>& _handle, const int _callbackType, EventDeligate& _functor);

                __host__ operator bool() const { return !handle.expired(); }
                __host__ bool operator <(const Listener& rhs) const { return hash < rhs.hash; }
            };

            std::atomic<int>                                            m_re;
            std::multimap<DirtinessEvent, Listener>                     m_listenerFromEvent;
            std::multimap<std::string, DirtinessEvent>                  m_eventFromAssetId;
            std::map<std::string, DirtinessEvent>                       m_eventHashFromId;
            
            enum GraphStates : uint      { kGraphUnlocked = 0, kGraphLocked };
            Semaphore                     m_semaphore;

        public:

            __host__ static DirtinessGraph& Get();

            __host__ bool AddListener(const std::string& id, WeakAssetHandle<Host::Asset>& handle, const DirtinessEvent& event, const int callbackType, EventDeligate functor = nullptr);
            __host__ void RemoveAllListeners(const std::string& assetId);
            __host__ void OnDirty(const DirtinessEvent& event, AssetHandle<Host::Asset>& caller);

            __host__ static DirtinessEvent RegisterEvent(const std::string& id, const bool mustExist);

        private:
            __host__ DirtinessGraph() : m_semaphore(kGraphUnlocked) {}                
        };

        class Dirtyable : public Host::Asset
        {
            friend DirtinessGraph;
        public:
            __host__ Dirtyable(const Asset::InitCtx& initCtx);
            __host__ virtual ~Dirtyable();

            __host__ bool               IsClean() const { return m_dirtyEvents.empty(); }
            __host__ bool               IsDirty(const DirtinessEvent& event) const;
            __host__ bool               IsAnyDirty(const std::vector<DirtinessEvent>& eventList) const;
            __host__ bool               IsAllDirty(const std::vector<DirtinessEvent>& eventList) const;
            __host__ void               Clean();

        protected:
            __host__ virtual void       OnDirty(const DirtinessEvent& event, AssetHandle<Host::Asset>& caller) {}
            __host__ virtual void       OnClean() {}

            __host__ void               SetDirty(const DirtinessEvent& event);
            __host__ void               SetDirty(const std::vector<DirtinessEvent>& eventList);
            __host__ void               UnsetDirty(const DirtinessEvent& event);
            __host__ void               UnsetDirty(const std::vector<DirtinessEvent>& eventList);

            //__host__ void               SignalDirty();
            __host__ void               SignalDirty(const DirtinessEvent& event);
            __host__ void               SignalDirty(const std::vector<DirtinessEvent>& eventList);

            __host__ void               Listen(const std::vector<DirtinessEvent>& eventList);
            __host__ void               Cascade(const std::vector<DirtinessEvent>& eventList);

            template<typename SuperClass, typename Deligate>
            __host__ __forceinline__ void Listen(const DirtinessEvent& event, SuperClass& super, Deligate deligate)
            {
                DirtinessGraph::EventDeligate functor(std::bind(deligate, &super, std::placeholders::_1, std::placeholders::_2));
                DirtinessGraph::Get().AddListener(GetAssetID(), GetAssetHandle(), event, kListenerDelegate, functor);
            }
        private:
            __host__ void               LockDirtyMutex();
            __host__ void               UnlockDirtyMutex();

        private:
            std::set<DirtinessEvent>        m_dirtyEvents;
            static std::recursive_mutex     m_dirtyMutex;
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