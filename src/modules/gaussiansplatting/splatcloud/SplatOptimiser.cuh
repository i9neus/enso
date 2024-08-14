#pragma once

#include "core/2d/Ctx.cuh"
#include "core/2d/DrawableObject.cuh"
#include "core/2d/RenderableObject.cuh"

#include "core/DirtinessFlags.cuh"
#include "core/Image.cuh"
#include "core/GenericObject.cuh"
#include "core/HighResolutionTimer.h"

#include "../FwdDecl.cuh"
#include "../scene/pointclouds/GaussianPointCloud.cuh"

namespace Enso
{             
    struct SplatOptimiserParams
    {
        __device__ void Validate() const {}
    };

    struct SplatOptimiserObjects
    {
        __device__ void Validate() const {}
    };

    namespace Host { class SplatOptimiser; }

    namespace Device
    {
        class SplatOptimiser : public Device::GenericObject
        {
            friend Host::SplatOptimiser;

        public:
            __host__ __device__ SplatOptimiser() {}

            __host__ __device__ void Synchronise(const SplatOptimiserParams& params) { m_params = params; }        
            __device__ void Synchronise(const SplatOptimiserObjects& objects) { m_objects = objects; }

        private:
            SplatOptimiserParams            m_params;
            SplatOptimiserObjects           m_objects;
        };
    }

    namespace Host
    {
        class SplatOptimiser : public Host::GenericObject
        {
        public:
            __host__                    SplatOptimiser(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ virtual            ~SplatOptimiser() noexcept;

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ static const std::string  GetAssetClassStatic() { return "splatoptimiser"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual void       Bind(GenericObjectContainer& objects) override final;
            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override final;

            __host__ Device::SplatOptimiser* GetDeviceInstance() const
            {
                return cu_deviceInstance;
            }

        private:
            Device::SplatOptimiser*           cu_deviceInstance = nullptr;
            SplatOptimiserObjects             m_objects;
            SplatOptimiserParams              m_params; 

            AssetHandle<Host::SceneContainer> m_hostSceneContainer;
            AssetHandle<Host::Camera>         m_hostActiveCamera;
        };
    }
}