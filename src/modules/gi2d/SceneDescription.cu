#include "SceneDescription.cuh"
#include "bih/BIH2DAsset.cuh"
#include "core/GenericObjectContainer.cuh"
#include "lights/Light.cuh"
#include "tracables/Tracable.cuh"
#include "integrators/Camera.cuh"

namespace Enso
{
    __host__ Host::SceneDescription::SceneDescription(const std::string& id) :
        Host::Asset(id),
        m_allocator(*this)
    {
        m_hostTracables = m_allocator.CreateChildAsset<Host::TracableContainer>("tracables", kVectorHostAlloc);
        m_hostLights = m_allocator.CreateChildAsset<Host::LightContainer>("lights", kVectorHostAlloc);
        m_hostCameras = m_allocator.CreateChildAsset<Host::CameraContainer>("cameras", kVectorHostAlloc);
        m_hostSceneObjects = m_allocator.CreateChildAsset<Host::SceneObjectContainer>("widgets", kVectorHostAlloc);

        m_hostTracableBIH = m_allocator.CreateChildAsset<Host::BIH2DAsset>("tracablebih", 1);
        m_hostSceneBIH = m_allocator.CreateChildAsset<Host::BIH2DAsset>("widgetbih", 1);

        m_deviceObjects.tracables = m_hostTracables->GetDeviceInstance();
        m_deviceObjects.lights = m_hostLights->GetDeviceInstance();
        m_deviceObjects.sceneObjects = m_hostSceneObjects->GetDeviceInstance();
        m_deviceObjects.tracableBIH = m_hostTracableBIH->GetDeviceInstance();
        m_deviceObjects.sceneBIH = m_hostSceneBIH->GetDeviceInstance();

        cu_deviceInstance = m_allocator.InstantiateOnDevice<Device::SceneDescription>();

        SynchroniseObjects<Device::SceneDescription>(cu_deviceInstance, m_deviceObjects);
    }

    __host__ void Host::SceneDescription::OnDestroyAsset()
    {
        m_allocator.DestroyOnDevice(cu_deviceInstance);

        m_hostTracables.DestroyAsset();
        m_hostLights.DestroyAsset();
        m_hostCameras.DestroyAsset();
        m_hostTracableBIH.DestroyAsset();
        m_hostSceneBIH.DestroyAsset();
    }

    __host__ Host::SceneDescription::~SceneDescription()
    {
        BEGIN_EXCEPTION_FENCE

            OnDestroyAsset();

        END_EXCEPTION_FENCE
    }

    template<typename ContainerType>
    __host__ void RebuildBIH(AssetHandle<Host::BIH2DAsset>& bih, ContainerType& primitives)
    {
        // Create a tracable list ready for building
        // TODO: It's probably faster if we build on the already-sorted index list
        auto& primIdxs = bih->GetPrimitiveIndices();
        primIdxs.resize(primitives->Size());
        for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

        // Construct the BIH
        std::function<BBox2f(uint)> getPrimitiveBBox = [&](const uint& idx) -> BBox2f
        {
            // Expand the world space bbox slightly
            return Grow((*primitives)[idx]->GetWorldSpaceBoundingBox(), 0.001f);
        };
        bih->Build(getPrimitiveBBox);
    }

    __host__ void Host::SceneDescription::Rebuild(AssetHandle<Host::GenericObjectContainer>& genericObjects, const UIViewCtx& viewCtx, const uint dirtyFlags)
    {        
        // Only rebuilid if the object bounds have changed through insertion, deletion or movement
        if (!(dirtyFlags & kDirtyIntegrators)) { return; }
        
        // Rebuild and synchronise any tracables that were dirtied since the last iteration
        int lightIdx = 0;
        m_hostTracables->Clear();
        m_hostLights->Clear();
        m_hostCameras->Clear();
        m_hostSceneObjects->Clear();

        genericObjects->ForEachOfType<Host::SceneObject>([&, this](AssetHandle<Host::SceneObject>& sceneObject) -> bool
            {
                // Rebuild the scene object (it will decide whether any action needs to be taken)
                if (sceneObject->Rebuild(dirtyFlags, viewCtx))
                {
                    // Ordinary objects go into the universal BIH
                    m_hostSceneObjects->EmplaceBack(sceneObject);

                    // Tracables have their own BIH and build process required for ray tracing, so process them separaately here.
                    auto tracable = sceneObject.DynamicCast<Host::Tracable>();
                    if (tracable)
                    {
                        // Add the tracable to the list
                        m_hostTracables->EmplaceBack(tracable);

                        // Tracables that are also lights are separated out into their own container
                        auto light = tracable.DynamicCast<Host::Light>();
                        if (light)
                        {
                            tracable->SetLightIdx(lightIdx++);
                            m_hostLights->EmplaceBack(light);
                        }
                    }
                }

                return true;
            }); 

        // Build a list of scene cameras for rendering
        genericObjects->ForEachOfType<Host::Camera>([&, this](AssetHandle<Host::Camera>& cameraObject) -> bool
            {
                if (cameraObject->Rebuild(dirtyFlags, viewCtx))
                {
                    m_hostCameras->EmplaceBack(cameraObject);
                }
            });

        // Sync the scene objects with the device
        m_hostTracables->Synchronise(kVectorSyncUpload);
        m_hostLights->Synchronise(kVectorSyncUpload);
        m_hostCameras->Synchronise(kVectorSyncUpload);
        m_hostSceneObjects->Synchronise(kVectorSyncUpload);

        // Rebuild the BIHs
        RebuildBIH(m_hostTracableBIH, m_hostTracables);
        RebuildBIH(m_hostSceneBIH, m_hostSceneObjects);

        {
            Log::Indent("Rebuild scene:");
            Log::Write("%i cameras", m_hostCameras->Size());
            Log::Write("Tracable BIH: %s", m_hostTracableBIH->GetBoundingBox().Format());
            Log::Write("Scene BIH: %s", m_hostSceneBIH->GetBoundingBox().Format());
        }

        // Cache the object bounding boxes
        /*m_tracableBBoxes.reserve(m_scene->m_hostTracables->Size());
        for (auto& tracable : *m_scene->m_hostTracables)
        {
            m_tracableBBoxes.emplace_back(tracable->GetBoundingBox());
        }*/        
    }
}