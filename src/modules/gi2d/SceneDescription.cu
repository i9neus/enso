#include "SceneDescription.cuh"
#include "bih/BIH2DAsset.cuh"
#include "integrators/VoxelProxyGrid.cuh"
#include "core/GenericObjectContainer.cuh"
#include "lights/OmniLight.cuh"

namespace Enso
{
    __host__ Host::SceneDescription::SceneDescription(const std::string& id) :
        Host::AssetAllocator(id)
    {
        m_hostTracables = CreateChildAsset<Host::TracableContainer>("tracables", kVectorHostAlloc);
        m_hostLights = CreateChildAsset<Host::LightContainer>("lights", kVectorHostAlloc);
        m_hostCameras = CreateChildAsset<Host::CameraContainer>("cameras", kVectorHostAlloc);
        m_hostWidgets = CreateChildAsset<Host::SceneObjectContainer>("widgets", kVectorHostAlloc);

        m_hostTracableBIH = CreateChildAsset<Host::BIH2DAsset>("tracablebih", 1);
        m_hostWidgetBIH = CreateChildAsset<Host::BIH2DAsset>("widgetbih", 1);

        m_deviceObjects.tracables = m_hostTracables->GetDeviceInstance();
        m_deviceObjects.lights = m_hostLights->GetDeviceInstance();
        m_deviceObjects.tracableBIH = m_hostTracableBIH->GetDeviceInstance();

        cu_deviceInstance = InstantiateOnDevice<Device::SceneDescription>();

        SynchroniseInheritedClass<Device::SceneDescription>(cu_deviceInstance, m_deviceObjects, 0);
    }

    __host__ void Host::SceneDescription::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceInstance);

        m_hostTracables.DestroyAsset();
        m_hostLights.DestroyAsset();
        m_hostCameras.DestroyAsset();
        m_hostTracableBIH.DestroyAsset();
        m_hostWidgetBIH.DestroyAsset();
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

    __host__ void Host::SceneDescription::Rebuild(AssetHandle<GenericObjectContainer>& renderObjects, const UIViewCtx& viewCtx, const uint dirtyFlags)
    {
        // Only rebuilid if the object bounds have change through insertion, deletion or movement
        if (!(dirtyFlags & kDirtyIntegrators)) { return; }
        
        // Rebuild and synchronise any tracables that were dirtied since the last iteration
        int lightIdx = 0;
        m_hostTracables->Clear();
        m_hostLights->Clear();

        renderObjects->ForEachOfType<Host::Tracable>([&, this](AssetHandle<Host::Tracable>& tracable) -> bool
            {
                // Rebuild the tracable (it will decide whether any action needs to be taken)
                if (tracable->Rebuild(dirtyFlags, viewCtx))
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

                return true;
            }); 

        // Make a list of drawable widgets that aren't tracables
        renderObjects->ForEach([&, this](AssetHandle<Host::GenericObject>& genericObject) -> bool
            {
                auto tracable = genericObject.DynamicCast<Host::Tracable>();
                if (!tracable)
                {
                    auto sceneObject = genericObject.DynamicCast<Host::SceneObject>();
                    if (sceneObject)
                    {
                        m_hostWidgets->EmplaceBack(sceneObject);
                    }
                }

                return true;
            });

        m_hostTracables->Synchronise(kVectorSyncUpload);
        m_hostLights->Synchronise(kVectorSyncUpload);
        m_hostCameras->Synchronise(kVectorSyncUpload);

        RebuildBIH(m_hostTracableBIH, m_hostTracables);
        Log::Write("Rebuilt scene BIH: %s", m_hostTracableBIH->GetBoundingBox().Format());

        // Cache the object bounding boxes
        /*m_tracableBBoxes.reserve(m_scene->m_hostTracables->Size());
        for (auto& tracable : *m_scene->m_hostTracables)
        {
            m_tracableBBoxes.emplace_back(tracable->GetBoundingBox());
        }*/

        
    }
}