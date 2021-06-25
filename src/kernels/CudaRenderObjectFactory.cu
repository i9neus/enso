#include "CudaRenderObjectFactory.cuh"
#include "CudaRenderObject.cuh"

#include "tracables/CudaKIFS.cuh"
#include "tracables/CudaSphere.cuh"
#include "tracables/CudaPlane.cuh"
#include "tracables/CudaCornell.cuh"

#include "lights/CudaQuadLight.cuh"
#include "lights/CudaEnvironmentLight.cuh"

#include "bxdfs/CudaLambert.cuh"

#include "materials/CudaMaterial.cuh"

namespace Cuda
{    
    RenderObjectFactory::RenderObjectFactory()
    {
        m_instantiators.emplace(Host::Sphere::GetAssetTypeString(), Host::Sphere::Instantiate);
        m_instantiators.emplace(Host::KIFS::GetAssetTypeString(), Host::KIFS::Instantiate);
        m_instantiators.emplace(Host::Plane ::GetAssetTypeString(), Host::Plane::Instantiate);
        m_instantiators.emplace(Host::Cornell::GetAssetTypeString(), Host::Cornell::Instantiate);

        m_instantiators.emplace(Host::QuadLight::GetAssetTypeString(), Host::QuadLight::Instantiate);
        m_instantiators.emplace(Host::EnvironmentLight::GetAssetTypeString(), Host::EnvironmentLight::Instantiate);

        m_instantiators.emplace(Host::SimpleMaterial::GetAssetTypeString(), Host::SimpleMaterial::Instantiate);

        m_instantiators.emplace(Host::LambertBRDF::GetAssetTypeString(), Host::LambertBRDF::Instantiate);
    }
   
    __host__ void RenderObjectFactory::InstantiateList(const Json::Node& node, const AssetType& expectedType, RenderObjectContainer& renderObjects)
    {
        for (Json::Node::ConstIterator it = node.begin(); it != node.end(); ++it)
        {
            std::string newType = it.Name();
            Json::Node childNode = *it;
            std::string newId;
            childNode.GetValue("id", newId, true);

            auto& instantiator = m_instantiators.find(newType);
            if (instantiator == m_instantiators.end())
            {
                Log::Error("Error: '%s' is not a valid render object type.", newType);
                continue;
            }

            if (renderObjects.Exists(newId))
            {
                Log::Error("Error: an object with ID '%s' has alread been instantiated.", newId);
            }

            AssetHandle<Host::RenderObject> newObject = (instantiator->second)(newId, expectedType, childNode);

            AssertMsgFmt(newObject, "The object instantiator for type '%s' did not return a valid render object.", newType.c_str());
        }
    }
    
    __host__ void RenderObjectFactory::Instantiate(const Json::Node& rootNode, RenderObjectContainer& renderObjects)
    {        
        const Json::Node tracablesNode = rootNode.GetChildObject("tracables", true);
        {
            InstantiateList
        }        
        
    }
}