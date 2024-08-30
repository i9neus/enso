#pragma once

#include "core/assets/GenericObject.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    namespace Device
    {
        class SceneObject : public Device::GenericObject
        {
        public:
            __device__ SceneObject() {}
            __device__ virtual ~SceneObject() noexcept {}
        };
    }
    
    namespace Host
    {      
        class SceneObject : public Host::GenericObject
        {
        public:
            __host__  SceneObject(const Asset::InitCtx& initCtx) :
                GenericObject(initCtx) {}

            __host__ virtual ~SceneObject() noexcept {}

            __host__ virtual void Bind(AssetHandle<Host::SceneContainer>& scene) {}   
        };
    }
}