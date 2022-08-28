﻿#pragma once

#include "AssetAllocator.cuh"
#include "math/CudaMath.cuh"
#include <map>
#include <string>
#include <functional>

namespace Json { class Node; }

namespace Cuda
{
    class RenderObjectContainer;
    namespace Host { class RenderObject; }
    
    class RenderObjectFactory
    {
        using InstantiatorLambda = std::function < AssetHandle<Host::RenderObject>(const std::string&, const AssetType&, const ::Json::Node&)>;
    public:
        __host__ RenderObjectFactory(cudaStream_t stream);

        __host__ void InstantiateSceneObjects(const ::Json::Node& json, AssetHandle<RenderObjectContainer>& renderObjects);
        __host__ void InstantiatePeripherals(const ::Json::Node& json, AssetHandle<RenderObjectContainer>& renderObjects);

    private:
        __host__ void InstantiateList(const ::Json::Node& node, const AssetType& assetType, const std::string& objectTypeStr, AssetHandle<RenderObjectContainer>& renderObjects);

        template<typename HostClass>
        __host__ void AddInstantiator()
        {
            const auto id = HostClass::GetAssetTypeString();
            auto it = m_instantiators.find(id);
            AssertMsgFmt(it == m_instantiators.end(),
                "Internal error: a render object instantiator with ID '%s' already exists.\n", id);

            m_instantiators[id] = HostClass::Instantiate;
            m_instanceFlagFunctors[id] = HostClass::GetInstanceFlags;
        }

        std::map<std::string, std::function<AssetHandle<Host::RenderObject>(const std::string&, const AssetType&, const ::Json::Node&)>>    m_instantiators;
        std::map<std::string, std::function<uint()>> m_instanceFlagFunctors;
        cudaStream_t m_hostStream;
    };
}