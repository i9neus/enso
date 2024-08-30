#pragma once

#include "Material.cuh"
#include "../../scene/SceneContainer.cuh"
#include "../textures/Texture2D.cuh"
#include "core/containers/Vector.cuh"
#include "core/math/ColourUtils.cuh"

namespace Enso
{
    __device__ Device::Material::Material() :
        m_textures(nullptr)
    {
    }

    __host__ Host::Material::Material(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene) :
        SceneObject(initCtx),
        cu_textures(scene->Textures().GetDeviceInstance())
    { 
    }

    __host__ void Host::Material::Bind(AssetHandle<Host::SceneContainer>& scene)
    {

    }

    __host__ void Host::Material::Synchronise(const uint syncFlags)
    {
        if (syncFlags & kSyncObjects)
        {
            SynchroniseTrivialParams(cu_deviceInstance, cu_textures);
        }
        OnSynchroniseMaterial(syncFlags);
    }
}