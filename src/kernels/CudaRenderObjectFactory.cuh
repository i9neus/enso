#pragma once

#include "math/CudaMath.cuh"
#include <map>

namespace Json { class Node; }

namespace Cuda
{
    class RenderObjectContainer;
    namespace Host { class RenderObject; }
    
    class RenderObjectFactory
    {
    public:
        __host__ RenderObjectFactory();

        __host__ void Instantiate(const ::Json::Node& json, AssetHandle<RenderObjectContainer>& renderObjects);

    private:
        __host__ void RenderObjectFactory::InstantiateList(const ::Json::Node& node, const AssetType& assetType, const std::string& objectTypeStr, AssetHandle<RenderObjectContainer>& renderObjects);

        std::map<std::string, std::function<AssetHandle<Host::RenderObject>(const std::string&, const AssetType&, const ::Json::Node&)>>    m_instantiators;
    };
}