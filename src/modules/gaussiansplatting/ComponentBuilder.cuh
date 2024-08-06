#pragma once

#include "ComponentContainer.cuh"
#include "core/DirtinessGraph.cuh"

namespace Enso
{
    namespace Host
    {
        class ComponentBuilder : public Host::Dirtyable
        {
        public:
            __host__                ComponentBuilder(const Asset::InitCtx& initCtx, AssetHandle<Host::ComponentContainer>& container);
            __host__ virtual        ~ComponentBuilder() noexcept {}

            __host__ void           Rebuild(bool forceRebuild);

            __host__ void           EnqueueEmplaceObject(AssetHandle<Host::GenericObject> handle);
            __host__ void           EnqueueDeleteObject(const std::string& assetId);

        protected:
            __host__ virtual void   OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller) override;

        private:
            __host__ void           SortDrawableObject(AssetHandle<Host::DrawableObject>& genericObject);

        private:
            AssetHandle<Host::ComponentContainer>                           m_container;
            
            std::unordered_map<void*, WeakAssetHandle<Host::Asset>>         m_rebuildQueue;
            std::unordered_set<std::string>                                 m_deleteQueue;
            std::vector<AssetHandle<Host::GenericObject>>                   m_emplaceQueue;

            int                                                             m_lightIdx;
            std::mutex                                                      m_mutex;
        };
    }
}