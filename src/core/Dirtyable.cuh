#include "math/Math.cuh"
#include "io/Serialisable.cuh"
#include "AssetAllocator.cuh"

#include <set>
#include <map>
#include <functional>
#include <mutex>

namespace Enso
{
    class GenericObject;
    class DirtinessGraph;
    class Dirtyable;

    using DirtinessKey = uint;

    namespace Host
    {
        class Dirtyable {};
        /*
        
        class Dirtyable
        {
            friend DirtinessGraph;
        public:
            __host__ Dirtyable(const std::string& id, DirtinessGraph& graph) : m_dirtinessGraph(graph) {}

            __host__ bool               IsDirty(const DirtinessKey& id);
            __host__ void               Clean();

        protected:
            __host__ virtual void       OnDirty(const DirtinessKey& id) = 0;
            __host__ void               SetDirty(const DirtinessKey& id);
            __host__ void               Listen(const DirtinessKey& id);

        private:
            std::set<DirtinessKey>      m_dirtyFlags;
            DirtinessGraph&             m_dirtinessGraph;
            Host::Asset&                m_parentAsset;
        };

        class DirtinessGraph
        {
        private:
            // TODO: use 
            std::multimap <DirtinessKey, WeakAssetHandle<Dirtyable>>    m_listeners;
            std::multimap <DirtinessKey, WeakAssetHandle<Dirtyable>>    m_eventQueue;

        public:
            __host__ DirtinessGraph() = default;

            __host__ void Flush();
            __host__ void AddListener(Dirtyable& owner, const DirtinessKey& eventID);
            __host__ void OnEvent(const DirtinessKey& flag);

            __host__ void Clear() { m_eventQueue.clear(); }
            __host__ void Flush();


        private:
            __host__ bool ListenerExists(Dirtyable& owner, const DirtinessKey& eventID) const;
        };*/
    }
}