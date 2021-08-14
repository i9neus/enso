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
        using InstantiatorLambda = std::function < AssetHandle<Host::RenderObject>(const std::string&, const AssetType&, const ::Json::Node&)>;
    public:
        __host__ RenderObjectFactory(cudaStream_t stream);

        __host__ void Instantiate(const ::Json::Node& json, AssetHandle<RenderObjectContainer>& renderObjects);

    private:
        __host__ void InstantiateList(const ::Json::Node& node, const AssetType& assetType, const std::string& objectTypeStr, AssetHandle<RenderObjectContainer>& renderObjects);
        
        __host__ void AddInstantiator(const std::string id, InstantiatorLambda& instantiator);

        std::map<std::string, std::function<AssetHandle<Host::RenderObject>(const std::string&, const AssetType&, const ::Json::Node&)>>    m_instantiators;
        cudaStream_t m_hostStream;
    };
}