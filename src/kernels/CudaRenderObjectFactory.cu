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

#include "CudaPerspectiveCamera.cuh"

#include "CudaWavefrontTracer.cuh"

namespace Cuda
{        
    RenderObjectFactory::RenderObjectFactory()
    {
        m_instantiators[Host::Sphere::GetAssetTypeString()] = Host::Sphere::Instantiate;
        m_instantiators[Host::KIFS::GetAssetTypeString()] = Host::KIFS::Instantiate;
        m_instantiators[Host::Plane::GetAssetTypeString()] = Host::Plane::Instantiate;
        //m_instantiators[Host::Cornell::GetAssetTypeString()] = Host::Cornell::Instantiate;\

        m_instantiators[Host::QuadLight::GetAssetTypeString()] = Host::QuadLight::Instantiate;
        m_instantiators[Host::EnvironmentLight::GetAssetTypeString()] = Host::EnvironmentLight::Instantiate;

        m_instantiators[Host::SimpleMaterial::GetAssetTypeString()] = Host::SimpleMaterial::Instantiate;

        m_instantiators[Host::LambertBRDF::GetAssetTypeString()] = Host::LambertBRDF::Instantiate;

        m_instantiators[Host::WavefrontTracer::GetAssetTypeString()] = Host::WavefrontTracer::Instantiate;

        m_instantiators[Host::PerspectiveCamera::GetAssetTypeString()] = Host::PerspectiveCamera::Instantiate;
    }
   
    __host__ void RenderObjectFactory::InstantiateList(const ::Json::Node& node, const AssetType& expectedType, const std::string& objectTypeStr, AssetHandle<RenderObjectContainer>& renderObjects)
    {
        for (::Json::Node::ConstIterator it = node.begin(); it != node.end(); ++it)
        {
            AssetHandle<Host::RenderObject> newObject;
            std::string newId = it.Name();
            ::Json::Node childNode = *it;
            std::string newClass;
            if (!childNode.GetValue("class", newClass, ::Json::kRequiredWarn)) { continue; }

            {
                Log::Indent indent(tfm::format("Creating new object '%s'...\n", newId));

                auto& instantiator = m_instantiators.find(newClass);
                if (instantiator == m_instantiators.end())
                {
                    Log::Error("Error: '%s' is not a valid render object type.\n", newClass);
                    continue;
                }

                Log::Debug("Instantiating new %s....\n", objectTypeStr);

                if (renderObjects->Exists(newId))
                {
                    Log::Error("Error: an object with ID '%s' has alread been instantiated.\n", newId);
                    continue;
                }

                newObject = (instantiator->second)(newId, expectedType, childNode);

                AssertMsgFmt(newObject, "The object instantiator for type '%s' did not return a valid render object.\n", newClass.c_str());

                renderObjects->Emplace(newObject);
            }
        }
    }
    
    __host__ void RenderObjectFactory::Instantiate(const ::Json::Node& rootNode, AssetHandle<RenderObjectContainer>& renderObjects)
    {               
        Assert(renderObjects);

        {
            const ::Json::Node childNode = rootNode.GetChildObject("tracables", ::Json::kRequiredAssert);
            InstantiateList(childNode, AssetType::kTracable, "tracable", renderObjects);
        }        
        {
            const ::Json::Node childNode = rootNode.GetChildObject("lights", ::Json::kRequiredAssert);
            InstantiateList(childNode, AssetType::kLight, "light", renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("materials", ::Json::kRequiredAssert);
            InstantiateList(childNode, AssetType::kMaterial, "material", renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("bxdfs", ::Json::kRequiredAssert);
            InstantiateList(childNode, AssetType::kBxDF, "BxDF", renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("cameras", ::Json::kRequiredAssert);
            InstantiateList(childNode, AssetType::kCamera, "camera", renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("integrators", ::Json::kRequiredAssert);
            InstantiateList(childNode, AssetType::kIntegrator, "integrator", renderObjects);
        }
    }
}