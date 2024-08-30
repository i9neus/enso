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

    __device__ vec3 Device::Material::EvaluateTexture(const vec2& uv, const int idx) const
    {
        return (idx >= 0 && idx < m_textures->size()) ? (*m_textures)[idx]->Evaluate(uv).xyz : kPink;
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