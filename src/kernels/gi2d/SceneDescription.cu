#include "SceneDescription.cuh"
#include "BIH2DAsset.cuh"
#include "integrators/VoxelProxyGrid.cuh"
#include "kernels/CudaRenderObjectContainer.cuh"
#include "lights/Light.cuh"

namespace GI2D
{
    __host__ Host::SceneDescription::SceneDescription(const std::string& id) :
        Cuda::Host::AssetAllocator(id)
    {
        m_hostTracables = CreateChildAsset<GI2D::Host::TracableContainer>("tracables", Core::kVectorHostAlloc);
        m_hostLights = CreateChildAsset<GI2D::Host::LightContainer>("lights", Core::kVectorHostAlloc);

        m_hostTracableBIH = CreateChildAsset<GI2D::Host::BIH2DAsset>("bih", 1);

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
        m_hostTracableBIH.DestroyAsset();
    }

    __host__ Host::SceneDescription::~SceneDescription()
    {
        BEGIN_EXCEPTION_FENCE

            OnDestroyAsset();

        END_EXCEPTION_FENCE
    }

    __host__ void Host::SceneDescription::Rebuild(Cuda::AssetHandle<Cuda::RenderObjectContainer>& renderObjects, const GI2D::UIViewCtx& viewCtx, const uint dirtyFlags)
    {
        // Rebuild and synchronise any tracables that were dirtied since the last iteration
        int lightIdx = 0;
        m_hostTracables->Clear();
        m_hostLights->Clear();

        renderObjects->ForEachOfType<Host::TracableInterface>([&, this](AssetHandle<GI2D::Host::TracableInterface>& tracable) -> bool
            {
                // Rebuild the tracable (it will decide whether any action needs to be taken)
                if (tracable->Rebuild(dirtyFlags, viewCtx))
                {
                    // Add the tracable to the list
                    m_hostTracables->EmplaceBack(tracable);
                    
                    // Tracables that are also lights are separated out into their own container
                    auto light = tracable.DynamicCast<GI2D::Host::LightInterface>();
                    if (light) 
                    { 
                        tracable->SetLightIdx(lightIdx++);
                        m_hostLights->EmplaceBack(light); 
                    }                    
                }

                return true;
            });       

        m_hostTracables->Synchronise(Core::kVectorSyncUpload);
        m_hostLights->Synchronise(Core::kVectorSyncUpload);

        // Cache the object bounding boxes
        /*m_tracableBBoxes.reserve(m_scene->m_hostTracables->Size());
        for (auto& tracable : *m_scene->m_hostTracables)
        {
            m_tracableBBoxes.emplace_back(tracable->GetBoundingBox());
        }*/

        // Create a tracable list ready for building
        // TODO: It's probably faster if we build on the already-sorted index list
        auto& primIdxs = m_hostTracableBIH->GetPrimitiveIndices();
        primIdxs.resize(m_hostTracables->Size());
        for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

        // Construct the BIH
        std::function<BBox2f(uint)> getPrimitiveBBox = [this](const uint& idx) -> BBox2f
        {
            return Grow((*m_hostTracables)[idx]->GetWorldSpaceBoundingBox(), 0.001f);
        };
        m_hostTracableBIH->Build(getPrimitiveBBox);
        //Log::Write("Rebuilt scene BIH: %s", m_scene->m_hostTracableBIH->GetBoundingBox().Format());
    }
}