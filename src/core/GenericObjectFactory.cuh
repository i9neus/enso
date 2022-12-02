#pragma once

#include "AssetAllocator.cuh"
#include "math/Math.cuh"
#include <map>
#include <string>
#include <functional>

namespace Enso
{
    namespace Json { class Node; }

    class GenericObjectContainer;
    namespace Host { class GenericObject; }
    
    class GenericObjectFactory
    {
        using InstantiatorLambda = std::function < AssetHandle<Host::GenericObject>(const std::string&, const int&, const Json::Node&)>;
    public:
        __host__ GenericObjectFactory();

        __host__ void InstantiateFromJson(const Json::Node& json, AssetHandle<GenericObjectContainer>& renderObjects);
        
        __host__ AssetHandle<Host::GenericObject> InstantiateFromHash(const uint hash, AssetHandle<GenericObjectContainer>& renderObjects);

        template<typename HostClass>
        __host__ void RegisterInstantiator(const uint hash = 0u)
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

        __host__ void RegisterGroup(const std::string& groupName, const uint flags);

    private:
        using InstantiatorSignature = AssetHandle<Host::GenericObject>(const std::string&, const Json::Node&);
        
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

        __host__ void InstantiateGroup(const Json::Node& node, const std::string& objectTypeStr, AssetHandle<GenericObjectContainer>& renderObjects);
        __host__ void EmplaceNewObject(AssetHandle<Host::GenericObject> newObject, AssetHandle<GenericObjectContainer>& renderObjects);

        std::map<std::string, std::shared_ptr<Instantiator>>    m_instantiators;
        std::map<uint, std::shared_ptr<Instantiator>>           m_hashMap;

        std::vector<ObjectGroup>                                m_groupList;
        int                                                     m_newInstanceCounter;
    };
}