#include "generic/JsonUtils.h"

#include "CudaRenderObjectFactory.cuh"
#include "CudaRenderObject.cuh"

#include "tracables/CudaKIFS.cuh"
#include "tracables/CudaSphere.cuh"
#include "tracables/CudaPlane.cuh"
//#include "tracables/CudaCornell.cuh"

#include "lights/CudaQuadLight.cuh"
#include "lights/CudaEnvironmentLight.cuh"

#include "bxdfs/CudaLambert.cuh"

#include "materials/CudaMaterial.cuh"

#include "CudaWavefrontTracer.cuh"

namespace Cuda
{        
    RenderObjectFactory::RenderObjectFactory()
    {
        m_instantiators[Host::Sphere::GetAssetTypeString()] = Host::Sphere::Instantiate;
        m_instantiators[Host::KIFS::GetAssetTypeString()] = Host::KIFS::Instantiate;
        m_instantiators[Host::Plane::GetAssetTypeString()] = Host::Plane::Instantiate;
        //m_instantiators[Host::Cornell::GetAssetTypeString()] = Host::Cornell::Instantiate;

        m_instantiators[Host::QuadLight::GetAssetTypeString()] = Host::QuadLight::Instantiate;
        m_instantiators[Host::EnvironmentLight::GetAssetTypeString()] = Host::EnvironmentLight::Instantiate;

        m_instantiators[Host::SimpleMaterial::GetAssetTypeString()] = Host::SimpleMaterial::Instantiate;

        m_instantiators[Host::LambertBRDF::GetAssetTypeString()] = Host::LambertBRDF::Instantiate;

        m_instantiators[Host::WavefrontTracer::GetAssetTypeString()] = Host::WavefrontTracer::Instantiate;
    }
   
    __host__ void RenderObjectFactory::InstantiateList(const ::Json::Node& node, const AssetType& expectedType, RenderObjectContainer& renderObjects)
    {
        for (::Json::Node::ConstIterator it = node.begin(); it != node.end(); ++it)
        {
            std::string newType = it.Name();
            ::Json::Node childNode = *it;
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

            renderObjects.Emplace(newObject);
        }
    }
    
    __host__ void RenderObjectFactory::Instantiate(const ::Json::Node& rootNode, RenderObjectContainer& renderObjects)
    {        
        {
            const ::Json::Node childNode = rootNode.GetChildObject("tracables", true);
            InstantiateList(childNode, AssetType::kTracable, renderObjects);
        }        
        {
            const ::Json::Node childNode = rootNode.GetChildObject("lights", true);
            InstantiateList(childNode, AssetType::kLight, renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("materials", true);
            InstantiateList(childNode, AssetType::kMaterial, renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("bxdfs", true);
            InstantiateList(childNode, AssetType::kBxDF, renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("cameras", true);
            InstantiateList(childNode, AssetType::kCamera, renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("integrators", true);
            InstantiateList(childNode, AssetType::kIntegrator, renderObjects);
        }
    }
}