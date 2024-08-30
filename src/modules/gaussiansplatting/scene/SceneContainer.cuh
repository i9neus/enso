#pragma once

#include "../FwdDecl.cuh"
#include "SceneObject.cuh"

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
                CudaAssert(lightSamplers);
                CudaAssert(materials);
                CudaAssert(textures);
                CudaAssert(cameras);
            }

            const Vector<Device::Tracable*>*    tracables = nullptr;
            const Vector<Device::LightSampler*>* lightSamplers = nullptr;
            const Vector<Device::Material*>*    materials = nullptr;
            const Vector<Device::Texture2D*>*   textures = nullptr;
            const Vector<Device::Camera*>*      cameras = nullptr;

            const Device::Texture2D*            envTexture = nullptr;
        };
    }

    namespace Host
    {
        class SceneBuilder;

        using TracableContainer = Host::AssetVector<Host::Tracable, Device::Tracable>;
        using LightSamplerContainer = Host::AssetVector<Host::LightSampler, Device::LightSampler>;
        using MaterialContainer = Host::AssetVector<Host::Material, Device::Material>;
        using Texture2DContainer = Host::AssetVector<Host::Texture2D, Device::Texture2D>;
        using CameraContainer = Host::AssetVector<Host::Camera, Device::Camera>;

        class SceneContainer : public Host::GenericObject
        {
        public:
            enum attrs_ : int { kInvalidContainerIdx = -1 };
        public:
            __host__                            SceneContainer(const Asset::InitCtx& initCtx);
            __host__ virtual                    ~SceneContainer() noexcept;     

            __host__ void                       Clean();            

            __host__ void                       DestroyManagedObjects();
            __host__ Device::SceneContainer*    GetDeviceInstance() const { return cu_deviceInstance; }
            __host__ const Device::SceneContainer& GetDeviceObjects() const { return m_deviceObjects; }

            __host__ virtual void               Synchronise(const uint flags) override final;

            __host__ void                       Summarise() const;
            __host__ static const std::string   GetAssetClassStatic() { return "scenecontainer"; }
            __host__ virtual std::string        GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ Host::TracableContainer&   Tracables() { Assert(m_hostTracables); return *m_hostTracables; }
            __host__ Host::LightSamplerContainer& LightSamplers() { Assert(m_hostLightSamplers); return *m_hostLightSamplers; }
            __host__ Host::MaterialContainer&   Materials() { Assert(m_hostMaterials); return *m_hostMaterials; }
            __host__ Host::Texture2DContainer&  Textures() { Assert(m_hostTextures); return *m_hostTextures; }
            __host__ Host::CameraContainer&     Cameras() { Assert(m_hostCameras); return *m_hostCameras; }

            __host__ void                       Emplace(AssetHandle<Host::Tracable> newTracable);
            __host__ void                       Emplace(AssetHandle<Host::LightSampler> newLight);
            __host__ void                       Emplace(AssetHandle<Host::Material> newMaterial);
            __host__ void                       Emplace(AssetHandle<Host::Texture2D> newTexture);
            __host__ void                       Emplace(AssetHandle<Host::Camera> newCamera);

            __host__ void                       SetEnvironmentTexture(const std::string& id);

            template<typename ObjectType>
            __host__ AssetHandle<ObjectType> Find(const std::string& id) const
            {
                auto it = m_sceneObjects.find(id);
                if (it != m_sceneObjects.end()) 
                { 
                    return it->second.DynamicCast<ObjectType>();
                }
                return nullptr;
            }

            __host__ int FindAssetIdx(const std::string& id) const
            {
                auto it = m_containerIndices.find(id);
                return (it != m_containerIndices.end()) ? it->second : kInvalidContainerIdx;
            }

        private:
            __host__ virtual void       OnDirty(const DirtinessEvent& flag, AssetHandle<Host::Asset>& caller) override final;
            __host__ virtual void       OnClean() override final;

        private:    
            // Vector containers for each type of scene object
            AssetHandle<Host::TracableContainer>    m_hostTracables;
            AssetHandle<Host::LightSamplerContainer> m_hostLightSamplers;
            AssetHandle<Host::MaterialContainer>    m_hostMaterials;
            AssetHandle<Host::Texture2DContainer>   m_hostTextures;
            AssetHandle<Host::CameraContainer>      m_hostCameras;

            // Map relating asset ID stems to their respective scene objects
            std::unordered_map<std::string, AssetHandle<Host::SceneObject>> m_sceneObjects;
             
            // Map relating asset ID stems to the indices in their respective containers
            std::unordered_map<std::string, int>    m_containerIndices;

            Device::SceneContainer*                 cu_deviceInstance = nullptr;
            Device::SceneContainer                  m_deviceObjects;

            std::unordered_map<std::string, WeakAssetHandle<Host::SceneObject>> m_syncObjectSet;

        };
    }
}