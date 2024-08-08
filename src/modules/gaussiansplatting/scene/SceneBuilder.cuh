#pragma once

#include "core/GenericObject.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    namespace Host
    {      
        class SceneBuilder : public Host::GenericObject
        {
            friend class SceneBuilder;

        public:
            __host__                SceneBuilder(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene);
            __host__ virtual        ~SceneBuilder() noexcept;

            __host__ virtual bool   Rebuild() override final;

        private:    
            AssetHandle<Host::SceneContainer>    m_scene;        
        };
    }
}