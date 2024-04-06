#include "GenericObjectContainer.cuh"

namespace Enso
{
    __host__ void GenericObjectContainer::Finalise() const
    {
        Log::Debug("Finalising...\n");
        Log::Indent indent;

        Log::Debug("DAG map:\n");
        {
            Log::Indent indent;
            for (auto& object : m_dagMap)
            {
                Log::Debug("%s\n", object.first);
            }
        }
    }

    __host__ void GenericObjectContainer::Emplace(AssetHandle<Host::GenericObject>& newObject, const bool requireDAGPath)
    {
        AssertMsgFmt(!Exists(newObject->GetAssetID()), "A render object with ID '%s' already exists in the object container.\n", newObject->GetAssetID().c_str());

        // Store a strong reference to the object in the object map
        m_objectMap[newObject->GetAssetID()] = newObject;        
        ++m_uniqueIdx;

        // If the object has a DAG path, add it to the map alongside its weak reference
        if (newObject->HasDAGPath())
        {
            if (m_dagMap.find(newObject->GetDAGPath()) == m_dagMap.end())
            {
                m_dagMap[newObject->GetDAGPath()] = newObject.GetWeakHandle();
            }
            else
            {
                Log::Error("Internal error: object '%s' has the same DAG path (%s) as another object.\n", newObject->GetAssetID(), newObject->GetDAGPath());
            }
        }
        // Child objects don't need to have DAG paths because they aren't user-referenceable
        else if (requireDAGPath && !newObject->IsChildObject())
        {
            Log::Warning("Warning: instantiated object '%s' does not have a valid DAG path. (Did you forget to call UpdateDAGPath() during FromJson()?)\n", newObject->GetAssetID());
            return;
        }
    }

    __host__ void GenericObjectContainer::Erase(const Host::GenericObject& obj)
    {
        Erase(obj.GetAssetID());
    }   

    __host__ void GenericObjectContainer::Erase(const std::string& id)
    {
        // Get the handle to the object
        auto it = m_objectMap.find(id);
        AssertMsgFmt(it != m_objectMap.end(), "Render object '%s' is not in the container.", id.c_str());
        auto obj = it->second;

        // Erase the object from the DAG map
        if (obj->HasDAGPath())
        {
            const auto& dag = obj->GetDAGPath();
            m_objectMap.erase(dag);
        }

        // Erase the object from the main asset map
        AssertMsgFmt(m_objectMap.erase(id), "Internal error: object map and object list have gone out of sync with object '%s'", id.c_str());

        // Destroy the asset
        obj.DestroyAsset();
    } 

    __host__ void GenericObjectContainer::Bind()
    {
        for (auto& object : m_objectMap)
        {
            object.second->Bind();
        }
    }

    __host__ void GenericObjectContainer::Synchronise()
    {
        for (auto& object : m_objectMap)
        {
            object.second->Synchronise();
        }
    }

    __host__ void GenericObjectContainer::OnDestroyAsset()
    {
        Log::Debug("Unloading scene graph...");
        
        constexpr int kMaxAttempts = 10;
        std::vector<std::string> activeList;
        for (int i = 0; !m_objectMap.empty() && i < kMaxAttempts; i++)
        {
            //Log::Indent indent(tfm::format("Pass %i...", i + 1));
            for (RenderObjectMap::iterator it = m_objectMap.begin(); it != m_objectMap.end();)
            {
                uint flags = kAssetCleanupPass;
                if (i == kMaxAttempts - 1)
                {
                    flags |= kAssetForceDestroy | kAssetAssertOnError;

                    if (it->second.GetReferenceCount() > 1)
                    {
                        activeList.push_back(it->first);
                    }
                }

                // Try to delete the asset
                if (!it->second.DestroyAsset(flags))
                {
                    ++it;
                }
                else
                {
                    auto nextIt = std::next(it);
                    m_objectMap.erase(it);
                    it = nextIt;
                }
            }
        }

        if (activeList.size() > 0)
        {
            Log::Error("ERROR: %i objects were not properly cleaned up:", activeList.size());
            for (const auto& name : activeList)
            {
                Log::Error("  - %s", name);
            }
        }
    }
}