#pragma once

#include "AssetAllocator.cuh"
#include "math/Math.cuh"
#include <map>
#include <string>
#include <functional>

#include "io/json/JsonUtils.h"
#include "GenericObjectContainer.cuh"
#include "GenericObject.cuh"

namespace Enso
{
    namespace Host
    {
        template<typename... TypePack>
        class GenericObjectFactory
        {
            using InstantiatorSignature = AssetHandle<Host::GenericObject>(const std::string&, TypePack...);
            //using InstantiatorLambda = std::function < AssetHandle<Host::GenericObject>(const std::string&, const int&, const Json::Node&)>;

            struct ObjectGroup
            {
                std::string     name;
                uint            flags;
            };

            struct Instantiator
            {
                std::string                             id;
                std::function<InstantiatorSignature>    instanceFunctor;
                std::function<uint()>                   flagFunctor;
            };

        public:
            __host__ GenericObjectFactory() : m_newInstanceCounter(0) {}

            //__host__ void                                   Instantiate(const Json::Node& rootNode, const AssetHandle<const Host::SceneContainer>&, Host::GenericObjectContainer& renderObjects);
            __host__ AssetHandle<Host::GenericObject>         Instantiate(const uint hash, Host::GenericObjectContainer& renderObjects, TypePack... pack);
            //__host__ AssetHandle<Host::GenericObject>       Instantiate(const std::string& id, const Json::Node& json, const AssetHandle<const Host::SceneContainer>&, Host::GenericObjectContainer& renderObjects);

            template<typename HostClass> __host__ void      RegisterInstantiator(const uint hash = 0u);

            __host__ void RegisterGroup(const std::string& groupName, const uint flags)
            {
                m_groupList.emplace_back(ObjectGroup{ groupName, flags });
            }

        private:
            __host__ AssetHandle<Host::GenericObject>       InstantiateImpl(std::shared_ptr<Instantiator>& ptr, Host::GenericObjectContainer& renderObjects, TypePack... pack);

            __host__ void InstantiateGroup(const Json::Node& node, const std::string& objectTypeStr, Host::GenericObjectContainer& renderObjects);
            __host__ void EmplaceNewObject(AssetHandle<Host::GenericObject> newObject, Host::GenericObjectContainer& renderObjects);

            std::map<std::string, std::shared_ptr<Instantiator>>    m_instantiators;
            std::map<uint, std::shared_ptr<Instantiator>>           m_hashMap;

            std::vector<ObjectGroup>                                m_groupList;
            int                                                     m_newInstanceCounter;
        };

        template<typename... TypePack>
        template<typename HostClass>
        __host__ void GenericObjectFactory<TypePack...>::RegisterInstantiator(const uint hash)
        {
            const auto id = HostClass::GetAssetClassStatic();
            auto it = m_instantiators.find(id);
            AssertMsgFmt(it == m_instantiators.end(),
                "Internal error: a render object instantiator with ID '%s' already exists.\n", id.c_str());

            auto newInst = std::make_shared<Instantiator>();
            m_instantiators[id] = newInst;
            newInst->id = id;
            newInst->instanceFunctor = HostClass::Instantiate;
            newInst->flagFunctor = HostClass::GetInstanceFlags;

            // If a hash has been specified, add it to the hash map
            if (hash != 0u)
            {
                auto it = m_hashMap.find(hash);
                AssertMsgFmt(it == m_hashMap.end(), "Internal error: a render object instantiator with hash %i already exists", hash);
                m_hashMap[hash] = newInst;
            }
        }

        template<typename... TypePack>
        __host__ void GenericObjectFactory<TypePack...>::InstantiateGroup(const Json::Node& node, const std::string& objectTypeStr, Host::GenericObjectContainer& renderObjects)
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

                        if (renderObjects.Exists(instanceId))
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

        template<typename... TypePack>
        __host__ void GenericObjectFactory<TypePack...>::EmplaceNewObject(AssetHandle<Host::GenericObject> newObject, Host::GenericObjectContainer& renderObjects)
        {
            renderObjects.Emplace(newObject);

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
                    renderObjects.Emplace(child);

                    Log::Debug("%s\n", child->GetAssetID());
                }
            }
        }

        /*template<typename... TypePack>
        __host__ void GenericObjectFactory<TypePack...>::Instantiate(const Json::Node& rootNode, const AssetHandle<const Host::SceneContainer>& scene,
            Host::GenericObjectContainer& renderObjects)
        {
            for (const auto& group : m_groupList)
            {
                const Json::Node childNode = rootNode.GetChildObject(group.name, group.flags);
                InstantiateGroup(childNode, group.name, scene, renderObjects);
            }
        }*/

        template<typename... TypePack>
        __host__ AssetHandle<Host::GenericObject> GenericObjectFactory<TypePack...>::Instantiate(const uint hash, Host::GenericObjectContainer& renderObjects, TypePack... pack)
        {
            // Look for an instantiator that matches the hash
            auto it = m_hashMap.find(hash);
            if (it == m_hashMap.end())
            {
                Log::Error("Error: an object with hash %i does not exist.", hash);
                return AssetHandle<Host::GenericObject>();
            }

            return InstantiateImpl(it->second, renderObjects, pack...);
        }

        /*template<typename... TypePack>
        __host__ AssetHandle<Host::GenericObject> GenericObjectFactory<TypePack...>::Instantiate(const std::string& id, TypePack... pack)
        {
            // Look for an instantiator that matches the hash
            auto it = m_instantiators.find(id);
            if (it == m_instantiators.end())
            {
                Log::Error("Error: an object with ID '%s' does not exist.", id);
                return AssetHandle<Host::GenericObject>();
            }

            return InstantiateImpl(it->second, json, scene, renderObjects);
        }*/

        template<typename... TypePack>
        __host__ AssetHandle<Host::GenericObject> GenericObjectFactory<TypePack...>::InstantiateImpl(std::shared_ptr<Instantiator>& instantiator, Host::GenericObjectContainer& renderObjects, TypePack... pack)
        {
            // Instantiate the object
            // FIXME: Create a better asset auto-naming system that this
            const std::string newAssetId = tfm::format("object%i", ++m_newInstanceCounter);
            AssetHandle<Host::GenericObject> newObject = instantiator->instanceFunctor(newAssetId, pack...);
            IsOk(cudaDeviceSynchronize());

            if (!newObject)
            {
                Log::Error("Failed to instantiate object of class '%s'.\n", instantiator->id);
                return AssetHandle<Host::GenericObject>();
            }

            // Emplace the newly created object
            EmplaceNewObject(newObject, renderObjects);

            return newObject;
        }
    }
}