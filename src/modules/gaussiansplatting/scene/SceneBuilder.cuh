#pragma once

#include "core/GenericObject.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    namespace Host
    {      
        class SceneBuilder
        {
        public:
            __host__                SceneBuilder() = default;

            __host__  bool          Rebuild(AssetHandle<Host::SceneContainer>& scene);
        };
    }
}