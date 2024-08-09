#pragma once

#include "../FwdDecl.cuh"
#include "core/GenericObject.cuh"

namespace Enso
{
    namespace Device
    { 
        struct SceneContainer
        {
            __device__ SceneContainer() {}
            __device__ void Synchronise(const SceneContainer& objects) { *this = objects; }
            
            __device__ void Validate() const
            {
                CudaAssert(tracables);
                CudaAssert(lights);
                CudaAssert(materials);
                CudaAssert(textures);
                CudaAssert(cameras);
            }

            const Vector<Device::Tracable*>*    tracables = nullptr;
            const Vector<Device::Light*>*       lights = nullptr;
            const Vector<Device::Material*>*    materials = nullptr;
            const Vector<Device::Texture2D*>*   textures = nullptr;
            const Vector<Device::Camera*>*      cameras = nullptr;
        };
    }

    namespace Host
    {
        class SceneBuilder;

        using TracableContainer = Host::AssetVector<Host::Tracable, Device::Tracable>;
        using LightContainer = Host::AssetVector<Host::Light, Device::Light>;
        using MaterialContainer = Host::AssetVector<Host::Material, Device::Material>;
        using Texture2DContainer = Host::AssetVector<Host::Texture2D, Device::Texture2D>;
        using CameraContainer = Host::AssetVector<Host::Camera, Device::Camera>;

        class SceneContainer : public Host::GenericObject
        {
        public:
            __host__                SceneContainer(const Asset::InitCtx& initCtx);
            __host__ virtual        ~SceneContainer() noexcept;     

            __host__ void           Prepare();
            __host__ void           Clean();

            __host__ void           DestroyManagedObjects();
            __host__ Device::SceneContainer* GetDeviceInstance() const { return cu_deviceInstance; }

            __host__ virtual void   Synchronise(const uint flags) override final;

            __host__ void           Summarise() const;
            __host__ static const std::string  GetAssetClassStatic() { return "scenecontainer"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            Host::TracableContainer&    Tracables() { Assert(m_hostTracables); return *m_hostTracables; }
            Host::LightContainer&       Lights() { Assert(m_hostLights); return *m_hostLights; }
            Host::MaterialContainer&    Materials() { Assert(m_hostMaterials); return *m_hostMaterials; }
            Host::Texture2DContainer&   Textures() { Assert(m_hostTextures); return *m_hostTextures; }
            Host::CameraContainer&      Cameras() { Assert(m_hostCameras); return *m_hostCameras; }

        private:    
            AssetHandle<Host::TracableContainer>    m_hostTracables;
            AssetHandle<Host::LightContainer>       m_hostLights;
            AssetHandle<Host::MaterialContainer>    m_hostMaterials;
            AssetHandle<Host::Texture2DContainer>   m_hostTextures;
            AssetHandle<Host::CameraContainer>      m_hostCameras;

            Device::SceneContainer*                 cu_deviceInstance = nullptr;
            Device::SceneContainer                  m_deviceObjects;
        };
    }
}