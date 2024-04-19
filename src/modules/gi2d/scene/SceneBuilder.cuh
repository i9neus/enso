#pragma once

#include "SceneContainer.cuh"
#include "core/DirtinessGraph.cuh"

namespace Enso
{
    namespace Host
    {
        class SceneBuilder : public Host::Dirtyable
        {
        public:
            __host__                SceneBuilder(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& container);
            __host__ virtual        ~SceneBuilder() noexcept {}

            __host__ void           Rebuild(const UIViewCtx& viewCtx, const uint dirtyFlags);

        protected:
            __host__ virtual void OnDirty(const DirtinessKey& flag, Host::Dirtyable& caller) override;

        private:
            AssetHandle<Host::SceneContainer>       m_container;
            AssetAllocator                          m_allocator;
        };
    }
}