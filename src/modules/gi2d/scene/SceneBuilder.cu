#include "SceneBuilder.cuh"
#include "../bih/BIH2DAsset.cuh"
#include "core/GenericObjectContainer.cuh"
#include "../lights/Light.cuh"
#include "../tracables/Tracable.cuh"
#include "../integrators/Camera.cuh"

namespace Enso
{    
    __host__ void Host::SceneBuilder::OnDirty(const DirtinessKey& flag, WeakAssetHandle<Host::Asset>& caller)
    {
        switch (flag)
        {
        case kDirtyObjectRebuild:
        case kDirtyObjectBoundingBox:
            {                                 
                // TODO: We're using the address of the object as the key. This probably isn't consistent with best practices. 
                m_rebuildQueue[(void*)caller.lock().get()] = caller;
                break;
            }
        };

        SetDirty(flag);
    }

    __host__ Host::SceneBuilder::SceneBuilder(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& container) :
        Dirtyable(initCtx),
        m_container(container)
    {
        Listen({ kDirtyObjectBoundingBox, kDirtyObjectRebuild, kDirtyObjectExistence });
    }

    template<typename ContainerType>
    __host__ void RebuildBIH(Host::BIH2DAsset& bih, ContainerType& primitives)
    {
        // Create a tracable list ready for building
        // TODO: It's probably faster if we build on the already-sorted index list
        auto& primIdxs = bih.GetPrimitiveIndices();
        primIdxs.Clear();
        primIdxs.Reserve(primitives.Size());
        Log::Debug("Building...");
        for (int idx = 0; idx < primitives.Size(); ++idx)
        { 
            // Ignore primitives that don't have bounding boxes
            if (primitives[idx]->HasBoundingBox())
            {
                primIdxs.PushBack(idx);
            }
        }

        // Construct the BIH
        std::function<BBox2f(uint)> getPrimitiveBBox = [&](const uint& idx) -> BBox2f
        {
            // Expand the world space bbox slightly
            return Grow(primitives[idx]->GetWorldSpaceBoundingBox(), 0.001f);
        };
        bih.Build(getPrimitiveBBox);
    }

    __host__ void Host::SceneBuilder::EnqueueEmplaceObject(AssetHandle<Host::GenericObject> handle)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_emplaceQueue.emplace_back(handle);
        SignalDirty(kDirtyObjectExistence);
    }

    __host__ void Host::SceneBuilder::EnqueueDeleteObject(const std::string& assetId)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_deleteQueue.emplace(assetId);
        SignalDirty(kDirtyObjectExistence);
    }

    __host__ void Host::SceneBuilder::SortSceneObject(AssetHandle<Host::SceneObject>& sceneObject)
    {
        // All scene objects go into the universal BIH
        m_container->m_hostSceneObjects->EmplaceBack(sceneObject);
        
        // If this is a camera object, add it to the list of cameras but not the list of scene objects
        auto camera = sceneObject.DynamicCast<Host::Camera>();
        if (camera)
        {
            m_container->m_hostCameras->EmplaceBack(camera);
        }
        else
        {
            // Tracables have their own BIH and build process required for ray tracing, so process them separaately here.
            auto tracable = sceneObject.DynamicCast<Host::Tracable>();
            if (tracable)
            {
                // Add the tracable to the list
                m_container->m_hostTracables->EmplaceBack(tracable);

                // Tracables that are also lights are separated out into their own container
                auto light = sceneObject.DynamicCast<Host::Light>();
                if (light)
                {
                    tracable->SetLightIdx(m_lightIdx++);
                    m_container->m_hostLights->EmplaceBack(light);
                }
            }
        }
    }

    __host__ void Host::SceneBuilder::Rebuild(bool forceRebuild)
    {                
        if (!forceRebuild && IsClean()) { return; } // Nothing to do!
        
        std::lock_guard<std::mutex> lock(m_mutex);
        m_lightIdx = 0;

        // If there are objects waiting in the delete queue, remove them now
        if (!m_deleteQueue.empty())
        {
            m_container->Clear();
            for (const auto& id : m_deleteQueue)
            {
                m_container->m_hostGenericObjects->Erase(id);
            }
            m_deleteQueue.clear();
        }

        // If there are objects waiting in the emplace queue, add them to the full list
        /*if (!m_emplaceQueue.empty())
        {
            for (auto& asset : m_emplaceQueue)
            {
                m_container->m_hostGenericObjects->Emplace(asset);
            }
            m_emplaceQueue.clear();
        }*/

        // If an object has been added or removed or a complete rebuild has been triggered, build the entire object container
        if (IsDirty(kDirtyObjectExistence) || forceRebuild)
        {
            m_container->Clear();
            for (auto& object : *m_container->m_hostGenericObjects)
            {
                auto sceneObject = object.DynamicCast<Host::SceneObject>();
                if (sceneObject)
                {
                    sceneObject->Rebuild();
                    SortSceneObject(sceneObject);
                }
            }
        }
        // Otherwise, only rebuild the scene objects listed in the rebuild queue
        else
        {
            for (auto& it : m_rebuildQueue)
            {
                if (it.second.expired())
                {
                    Log::Error("Warning: an object expired from the rebuild queue before it could be rebuilt");
                }
                else
                {
                    AssetHandle<Host::SceneObject> sceneObject = AssetHandle<Host::Asset>(it.second).DynamicCast<Host::SceneObject>();
                    if (sceneObject)
                    {
                        sceneObject->Rebuild();
                    }
                }
            }
        }

        // Synchronise the container lists
        m_container->Synchronise();

        // TODO: Make this process more intelligent by only doing a full rebuild if any object bounding boxes have explicitly changed
        if (IsAnyDirty({ kDirtyObjectBoundingBox, kDirtyObjectExistence }))
        {
            // Rebuild the BIHs
            RebuildBIH(*m_container->m_hostTracableBIH, *m_container->m_hostTracables);
            RebuildBIH(*m_container->m_hostSceneBIH, *m_container->m_hostSceneObjects);
        }

        SignalDirty(kDirtyIntegrators);

        // Summarise the build process
        //m_container->Summarise();

        // Clean up
        m_rebuildQueue.clear();
        Clean();
    }
}