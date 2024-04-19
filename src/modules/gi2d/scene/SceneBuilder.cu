#include "SceneBuilder.cuh"
#include "../bih/BIH2DAsset.cuh"
#include "core/GenericObjectContainer.cuh"
#include "../lights/Light.cuh"
#include "../tracables/Tracable.cuh"
#include "../integrators/Camera.cuh"

namespace Enso
{
    __host__ void Host::SceneBuilder::OnDirty(const DirtinessKey& flag, Host::Dirtyable& caller)
    {
        if (flag == 1u)
        {
            Log::Error("Caught!");
        }
    }

    __host__ Host::SceneBuilder::SceneBuilder(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& container) :
        Dirtyable(initCtx),
        m_allocator(*this),
        m_container(container)
    {
    }

    template<typename ContainerType>
    __host__ void RebuildBIH(Host::BIH2DAsset& bih, ContainerType& primitives)
    {
        // Create a tracable list ready for building
        // TODO: It's probably faster if we build on the already-sorted index list
        auto& primIdxs = bih.GetPrimitiveIndices();
        primIdxs.resize(primitives.Size());
        for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

        // Construct the BIH
        std::function<BBox2f(uint)> getPrimitiveBBox = [&](const uint& idx) -> BBox2f
        {
            // Expand the world space bbox slightly
            return Grow(primitives[idx]->GetWorldSpaceBoundingBox(), 0.001f);
        };
        bih.Build(getPrimitiveBBox);
    }

    __host__ void Host::SceneBuilder::Rebuild(const UIViewCtx& viewCtx, const uint dirtyFlags)
    {
        // Only rebuilid if the object bounds have changed through insertion, deletion or movement
        if (!(dirtyFlags & kDirtyIntegrators)) { return; }

        // Rebuild and synchronise any tracables that were dirtied since the last iteration
        int lightIdx = 0;

        auto& cameras = *m_container->m_hostCameras;
        auto& lights = *m_container->m_hostLights;
        auto& tracables = *m_container->m_hostTracables;
        auto& sceneObjects = *m_container->m_hostSceneObjects;
        auto& genericObjects = *m_container->m_hostGenericObjects;

        m_container->Clear();

        genericObjects.ForEach([&, this](AssetHandle<Host::GenericObject>& genericObject) -> bool
            {
                // If this is a camera object, add it to the list of cameras
                auto camera = genericObject.DynamicCast<Host::Camera>();
                if (camera)
                {
                    if (camera->Rebuild(dirtyFlags, viewCtx))
                    {
                        cameras.EmplaceBack(camera);
                    }
                    return true;
                }

                auto sceneObject = genericObject.DynamicCast<Host::SceneObject>();
                // Rebuild the scene object (it will decide whether any action needs to be taken)
                if (sceneObject && sceneObject->Rebuild(dirtyFlags, viewCtx))
                {
                    // If the object can be transformed, add it to the list of scene objects
                    if (sceneObject->IsTransformable())
                    {
                        // Ordinary objects go into the universal BIH
                        sceneObjects.EmplaceBack(sceneObject);

                        // Tracables have their own BIH and build process required for ray tracing, so process them separaately here.
                        auto tracable = genericObject.DynamicCast<Host::Tracable>();
                        if (tracable)
                        {
                            // Add the tracable to the list
                            tracables.EmplaceBack(tracable);

                            // Tracables that are also lights are separated out into their own container
                            auto light = genericObject.DynamicCast<Host::Light>();
                            if (light)
                            {
                                tracable->SetLightIdx(lightIdx++);
                                lights.EmplaceBack(light);
                            }
                        }
                    }
                }

                return true;
            });

        // Synch
        m_container->Synchronise();

        // Rebuild the BIHs
        RebuildBIH(*m_container->m_hostTracableBIH, tracables);
        RebuildBIH(*m_container->m_hostSceneBIH, sceneObjects);

        m_container->Summarise();

        // Cache the object bounding boxes
        /*m_tracableBBoxes.reserve(m_scene->m_hostTracables->Size());
        for (auto& tracable : *m_scene->m_hostTracables)
        {
            m_tracableBBoxes.emplace_back(tracable->GetBoundingBox());
        }*/
    }
}