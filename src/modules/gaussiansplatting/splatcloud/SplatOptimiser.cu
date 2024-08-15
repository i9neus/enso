#include "SplatOptimiser.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/3d/Ctx.cuh"
#include "core/assets/AssetContainer.cuh"
#include "core/3d/Transform.cuh"
#include "core/containers/Vector.cuh"
#include "core/3d/Cameras.cuh"
#include "core/assets/GenericObjectContainer.cuh"

#include "../scene/SceneContainer.cuh"
#include "../scene/cameras/Camera.cuh"
#include "../scene/materials/Material.cuh"
#include "../scene/lights/Light.cuh"
#include "../scene/textures/Texture2D.cuh"
#include "../scene/tracables/Tracable.cuh"

#include "io/json/JsonUtils.h"
//#include "core/AccumulationBuffer.cuh"

namespace Enso
{        
     __host__ AssetHandle<Host::GenericObject> Host::SplatOptimiser::Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects)
    {
        return AssetAllocator::CreateChildAsset<Host::SplatOptimiser>(parentAsset, id, genericObjects);
    }

    __host__ Host::SplatOptimiser::SplatOptimiser(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects) :
        GenericObject(initCtx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::SplatOptimiser>(*this))
    {                
        
    }

    __host__ Host::SplatOptimiser::~SplatOptimiser() noexcept
    {      
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }   

    __host__ void Host::SplatOptimiser::Bind(GenericObjectContainer& objects)
    {
      

        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ bool Host::SplatOptimiser::Serialise(Json::Node& node, const int flags) const
    {
        GenericObject::Serialise(node, flags);

        return true;
    }

    __host__ bool Host::SplatOptimiser::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = GenericObject::Deserialise(node, flags);

        if (isDirty) { SignalDirty({ kDirtyParams }); }

        return isDirty;
    }

}