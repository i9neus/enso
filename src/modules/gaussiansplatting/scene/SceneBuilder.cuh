#pragma once

#include "core/assets/GenericObject.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    namespace Host
    {      
        class SceneBuilder
        {
        public:
            __host__                SceneBuilder();

            __host__  bool          Rebuild(AssetHandle<Host::SceneContainer>& scene);

        private:
            //__host__ virtual void   OnDirty(const DirtinessEvent& flag, AssetHandle<Host::Asset>& caller) override final;
        };
    }
}