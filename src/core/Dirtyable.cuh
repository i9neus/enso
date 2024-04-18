#include "math/Math.cuh"
#include "io/Serialisable.cuh"
#include "AssetAllocator.cuh"

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
        
        class Dirtyable : public Host::Asset
        {
            friend DirtinessGraph;
        public:
            __host__ Dirtyable(const Asset::InitCtx& initCtx);

            __host__ bool               IsDirty(const DirtinessKey& id);
            __host__ void               Clean();

        protected:
            __host__ virtual void       OnDirty(const DirtinessKey& id, const WeakAssetHandle<Dirtyable>& asset) {}
            __host__ void               SetDirty(const DirtinessKey& id);
            __host__ void               Listen(const DirtinessKey& id);

        private:
            std::set<DirtinessKey>      m_dirtyFlags;
        };

        class DirtinessGraph
        {
        private:           
            std::mutex                                                  m_mutex;
            std::multimap <DirtinessKey, WeakAssetHandle<Dirtyable>>    m_listenerHandles;            
            std::multimap <std::string, DirtinessKey>                   m_listenerKeys;
            std::multimap <DirtinessKey, WeakAssetHandle<Dirtyable>>    m_eventQueue;

        public:
            __host__ DirtinessGraph() = default;

            __host__ bool AddListener(const std::string& id, WeakAssetHandle<Host::Dirtyable>& handle, const DirtinessKey& eventId);
            __host__ void OnEvent(const DirtinessKey& flag);

            __host__ void Clear() { m_eventQueue.clear(); }
            __host__ void Flush();

        private:
            __host__ bool ListenerExists(Dirtyable& owner, const DirtinessKey& eventID) const;
            __host__ void GarbageCollect();
        };

    }
}