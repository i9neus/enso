#pragma once

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
            __host__ virtual void       OnDirty(const DirtinessKey& id, Host::Dirtyable& caller) {}
            __host__ void               SetDirty(const DirtinessKey& id);
            __host__ void               SignalDirty(const DirtinessKey& id);
            __host__ void               Listen(const DirtinessKey& id);

        private:
            std::set<DirtinessKey>      m_dirtyFlags;
        };

        class DirtinessGraph
        {
        private:           
            std::mutex                                                  m_mutex;
            std::multimap <DirtinessKey, WeakAssetHandle<Dirtyable>>    m_handleFromFlag;            
            std::multimap <std::string, DirtinessKey>                   m_flagFromAssetId;
            std::set<DirtinessKey>                                      m_eventSet;

        public:
            __host__ DirtinessGraph() = default;

            __host__ bool AddListener(const std::string& id, WeakAssetHandle<Host::Dirtyable>& handle, const DirtinessKey& flag);
            __host__ void OnDirty(const DirtinessKey& flag, Host::Dirtyable& caller);

            __host__ void Clear() { m_eventSet.clear(); }
            __host__ void Flush();
        };

    }
}