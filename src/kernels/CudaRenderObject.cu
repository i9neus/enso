#include "CudaRenderObject.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ void Host::RenderObject::UpdateDAGPath(const ::Json::Node& node)
    {
        if (!GetDAGPath().empty()) { return; }
        
        if (node.HasDAGPath())
        {
            SetDAGPath(node.GetDAGPath());
        }
        else
        {
            Log::Error("Internal error: JSON node for '%s' has no DAG path.\n", GetAssetID());
        }       
    }

    __host__ void RenderObjectContainer::Finalise() const
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

    __host__ void RenderObjectContainer::Emplace(AssetHandle<Host::RenderObject>& newObject)
    {
        AssertMsgFmt(!Exists(newObject->GetAssetID()), "A render object with ID '%s' already exists in the object container.\n", newObject->GetAssetID().c_str());

        m_objectMap[newObject->GetAssetID()] = newObject;

        if (newObject->HasDAGPath())
        {
            if (m_dagMap.find(newObject->GetDAGPath()) == m_dagMap.end())
            {
                m_dagMap[newObject->GetDAGPath()] = newObject;
            }
            else
            {
                Log::Error("Internal error: object '%s' has the same DAG path (%s) as another object.\n", newObject->GetAssetID(), newObject->GetDAGPath());
            }
        }
        else if(!newObject->IsChildObject())
        {
            Log::Error("Error: instantiated object '%s' does not have a valid DAG path. (Did you forget to call UpdateDAGPath() during FromJson()?)\n", newObject->GetAssetID());
            return;
        }
    }

    __host__ void RenderObjectContainer::Bind()
    {
        for (auto& object : m_objectMap)
        {
            object.second->Bind(*this);
        }
    }

    __host__ void RenderObjectContainer::Synchronise()
    {
        for (auto& object : m_objectMap)
        {
            object.second->Synchronise();
        }
    }

    __host__ void RenderObjectContainer::OnDestroyAsset()
    {
        constexpr int kMaxAttempts = 10;
        for (int i = 0; !m_objectMap.empty() && i < kMaxAttempts; i++)
        {
            for (RenderObjectMap::iterator it = m_objectMap.begin(); it != m_objectMap.end();)
            {
                uint flags = 0;
                if (i == kMaxAttempts - 1)
                {
                    flags |= kAssetForceDestroy | kAssetAssertOnError;
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
    }
}