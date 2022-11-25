#include "io/json/JsonUtils.h"

#include "GenericObjectFactory.cuh"
#include "GenericObjectContainer.cuh"

namespace Enso
{            
    GenericObjectFactory::GenericObjectFactory() :
        m_newInstanceCounter(0)
    {
        
    }

    __host__ void GenericObjectFactory::RegisterGroup(const std::string& groupName, const uint flags)
    {
        m_groupList.emplace_back(ObjectGroup{ groupName, flags });
    }
   
    __host__ void GenericObjectFactory::InstantiateGroup(const Json::Node& node, const std::string& objectTypeStr, AssetHandle<GenericObjectContainer>& renderObjects)
    {
        for (Json::Node::ConstIterator it = node.begin(); it != node.end(); ++it)
        {
            AssetHandle<Host::GenericObject> newObject;
            std::string newId = it.Name();
            Json::Node childNode = *it;

            if (newId.empty())
            {
                Log::Warning("Warning: skipping object with empty ID.\n");
                continue;
            }

            if (!childNode.GetBool("enabled", true, Json::kSilent)) { continue; }

            std::string newClass;
            if (!childNode.GetValue("class", newClass, Json::kRequiredWarn)) { continue; }

            {
                Log::Indent indent(tfm::format("Creating new object '%s'...\n", newId));

                const auto& it = m_instantiators.find(newClass);
                if (it == m_instantiators.end())
                {
                    Log::Error("Error: '%s' is not a valid render object type.\n", newClass);
                    continue;
                }
                const auto instantiator = *(it->second);

                // Get the object instance flags                
                const uint instanceFlags = instantiator.flagFunctor();

                int numInstances = 1;
                if (childNode.GetValue("instances", numInstances, Json::kSilent))
                {
                    // Check if this class allows for multiple instances from the same object
                    if (!(instanceFlags & kInstanceFlagsAllowMultipleInstances))
                    {
                        numInstances = 1;
                        Log::Warning("Warning: render objects of type '%s' do not allow multiple instantiation.\n", newClass);
                    }
                    else if (numInstances < 1 || numInstances > 10)
                    {
                        Log::Warning("Warning: instances out of range. Resetting to 1.\n");
                    }
                }

                Log::Debug("Instantiating %i new %s....\n", numInstances, objectTypeStr);

                std::string instanceId;
                for (int instanceIdx = 0; instanceIdx < numInstances; ++instanceIdx)
                {
                    // If the ID is a number, append it with an underscore to avoid breaking the DAG convention
                    const std::string instanceId = (numInstances == 1) ? newId : tfm::format("%s%i", newId, instanceIdx + 1);

                    if (renderObjects->Exists(instanceId))
                    {
                        Log::Error("Error: an object with ID '%s' has alread been instantiated.\n", instanceId);
                        continue;
                    }

                    // Instantiate the object
                    newObject = instantiator.instanceFunctor(instanceId, childNode);
                    IsOk(cudaDeviceSynchronize());

                    if (!newObject)
                    {
                        Log::Error("Failed to instantiate object '%s' of class '%s'.\n", instanceId, newClass);
                        continue;
                    }

                    // Instanced objects have virtual DAG paths, so replace the trailing ID from the JSON file with the actual ID from the asset
                    const std::string instancedDAGPath = tfm::format("%s%c%i", childNode.GetDAGPath(), Json::Node::kDAGDelimiter, instanceIdx + 1);
                    newObject->SetDAGPath(instancedDAGPath);

                    // Emplace the newly created object
                    EmplaceNewObject(newObject, renderObjects);
                }
            }
        }
    }

    __host__ void GenericObjectFactory::EmplaceNewObject(AssetHandle<Host::GenericObject> newObject, AssetHandle<GenericObjectContainer>& renderObjects)
    {
        renderObjects->Emplace(newObject);

        // The render object may have generated some of its own assets. Add them to the object list. 
        std::vector<AssetHandle<Host::GenericObject>> childObjects = newObject->GetChildObjectHandles();
        if (!childObjects.empty())
        {
            Log::Debug("Captured %i child objects:\n", childObjects.size());
            Log::Indent indent2;
            for (auto& child : childObjects)
            {
                AssertMsg(child, "A captured child object handle is invalid.");
                //child->SetHostStream(m_hostStream);
                renderObjects->Emplace(child);

                Log::Debug("%s\n", child->GetAssetID());
            }
        }
    }

    __host__ void GenericObjectFactory::InstantiateFromJson(const Json::Node& rootNode, AssetHandle<GenericObjectContainer>& renderObjects)
    {
        for (const auto& group : m_groupList)
        {
            const Json::Node childNode = rootNode.GetChildObject(group.name, group.flags);
            InstantiateGroup(childNode, group.name, renderObjects);
        }
    }  

    __host__ AssetHandle<Host::GenericObject> GenericObjectFactory::InstantiateFromHash(const uint hash, AssetHandle<GenericObjectContainer>& renderObjects)
    {
        // Look for an instantiator that matches the hash
        auto it = m_hashMap.find(hash);
        if (it == m_hashMap.end()) 
        { 
            Log::Error("Error: an object with hash %i does not exist.", hash);
            return AssetHandle<Host::GenericObject>();
        }
        auto instantiator = *it->second;

        // Instantiate the object
        Json::Document childNode;
        AssetHandle<Host::GenericObject> newObject = instantiator.instanceFunctor(tfm::format("object%i", ++m_newInstanceCounter), childNode);
        IsOk(cudaDeviceSynchronize());

        if (!newObject)
        {
            Log::Error("Failed to instantiate object of class '%s'.\n", instantiator.id);
            return AssetHandle<Host::GenericObject>();
        }

        // Emplace the newly created object
        EmplaceNewObject(newObject, renderObjects);

        return newObject;
    }
}