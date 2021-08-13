﻿#include "generic/JsonUtils.h"

#include "CudaRenderObjectFactory.cuh"
#include "CudaRenderObject.cuh"

#include "tracables/CudaKIFS.cuh"
#include "tracables/CudaSphere.cuh"
#include "tracables/CudaPlane.cuh"
#include "tracables/CudaCornellBox.cuh"

#include "lights/CudaQuadLight.cuh"
#include "lights/CudaEnvironmentLight.cuh"

#include "bxdfs/CudaLambert.cuh"

#include "materials/CudaSimpleMaterial.cuh"
#include "materials/CudaCornellMaterial.cuh"

#include "cameras/CudaPerspectiveCamera.cuh"
#include "cameras/CudaLightProbeCamera.cuh"
#include "cameras/CudaFisheyeCamera.cuh"

#include "CudaWavefrontTracer.cuh"

namespace Cuda
{        
    RenderObjectFactory::RenderObjectFactory(cudaStream_t hostStream) :
        m_hostStream(hostStream) 
    {
        m_instantiators[Host::Sphere::GetAssetTypeString()] = Host::Sphere::Instantiate;
        m_instantiators[Host::KIFS::GetAssetTypeString()] = Host::KIFS::Instantiate;
        m_instantiators[Host::Plane::GetAssetTypeString()] = Host::Plane::Instantiate;
        m_instantiators[Host::CornellBox::GetAssetTypeString()] = Host::CornellBox::Instantiate;

        m_instantiators[Host::QuadLight::GetAssetTypeString()] = Host::QuadLight::Instantiate;
        m_instantiators[Host::EnvironmentLight::GetAssetTypeString()] = Host::EnvironmentLight::Instantiate;

        m_instantiators[Host::SimpleMaterial::GetAssetTypeString()] = Host::SimpleMaterial::Instantiate;
        m_instantiators[Host::CornellMaterial::GetAssetTypeString()] = Host::CornellMaterial::Instantiate;

        m_instantiators[Host::LambertBRDF::GetAssetTypeString()] = Host::LambertBRDF::Instantiate;

        m_instantiators[Host::WavefrontTracer::GetAssetTypeString()] = Host::WavefrontTracer::Instantiate;

        m_instantiators[Host::PerspectiveCamera::GetAssetTypeString()] = Host::PerspectiveCamera::Instantiate;
        m_instantiators[Host::LightProbeCamera::GetAssetTypeString()] = Host::LightProbeCamera::Instantiate;
        m_instantiators[Host::FisheyeCamera::GetAssetTypeString()] = Host::FisheyeCamera::Instantiate;
    }
   
    __host__ void RenderObjectFactory::InstantiateList(const ::Json::Node& node, const AssetType& expectedType, const std::string& objectTypeStr, AssetHandle<RenderObjectContainer>& renderObjects)
    {
        for (::Json::Node::ConstIterator it = node.begin(); it != node.end(); ++it)
        {
            AssetHandle<Host::RenderObject> newObject;
            std::string newId = it.Name();
            ::Json::Node childNode = *it;

            if (!childNode.GetBool("enabled", true, ::Json::kSilent)) { continue; }

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
                if (!newObject)
                {
                    Log::Error("Object class '%s' is not a member of group '%s'. Move it to the correct group and try again.\n", newClass.c_str(), objectTypeStr);
                    continue;
                }

                // Emplace the newly created object
                newObject->SetHostStream(m_hostStream);
                renderObjects->Emplace(newObject);

                // The render object may have generated some of its own assets. Add them to the object list. 
                std::vector<AssetHandle<Host::RenderObject>> childObjects = newObject->GetChildObjectHandles();
                if(!childObjects.empty())
                {
                    Log::Debug("Captured %i child objects:\n", childObjects.size());
                    Log::Indent indent2;
                    for (auto& child : childObjects)
                    {
                        Assert(child);
                        child->SetHostStream(m_hostStream);
                        renderObjects->Emplace(child);

                        Log::Debug("%s\n", child->GetAssetID());
                    }
                }
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