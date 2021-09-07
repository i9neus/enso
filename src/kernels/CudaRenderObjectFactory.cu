#include "generic/JsonUtils.h"

#include "CudaRenderObjectFactory.cuh"
#include "CudaRenderObject.cuh"

#include "tracables/CudaKIFS.cuh"
#include "tracables/CudaSphere.cuh"
#include "tracables/CudaPlane.cuh"
#include "tracables/CudaCornellBox.cuh"
#include "tracables/CudaBox.cuh"

#include "lights/CudaQuadLight.cuh"
#include "lights/CudaSphereLight.cuh"
#include "lights/CudaEnvironmentLight.cuh"
#include "lights/CudaDistantLight.cuh"

#include "bxdfs/CudaLambert.cuh"

#include "materials/CudaSimpleMaterial.cuh"
#include "materials/CudaCornellMaterial.cuh"
#include "materials/CudaKIFSMaterial.cuh"

#include "cameras/CudaPerspectiveCamera.cuh"
#include "cameras/CudaLightProbeCamera.cuh"

#include "lightprobes/CudaLightProbeKernelFilter.cuh"
#include "lightprobes/CudaLightProbeRegressionFilter.cuh"

#include "CudaWavefrontTracer.cuh"

namespace Cuda
{            
    RenderObjectFactory::RenderObjectFactory(cudaStream_t hostStream) :
        m_hostStream(hostStream) 
    {
        AddInstantiator<Host::Sphere>();
        AddInstantiator<Host::KIFS>();
        AddInstantiator<Host::Plane>();
        AddInstantiator<Host::CornellBox>();
        AddInstantiator<Host::Box>();

        AddInstantiator<Host::QuadLight>();
        AddInstantiator<Host::SphereLight>();
        AddInstantiator<Host::EnvironmentLight>();
        AddInstantiator<Host::DistantLight>();

        AddInstantiator<Host::SimpleMaterial>();
        AddInstantiator<Host::CornellMaterial>();
        AddInstantiator<Host::KIFSMaterial>();

        AddInstantiator<Host::LambertBRDF>();

        AddInstantiator<Host::WavefrontTracer>();

        AddInstantiator<Host::PerspectiveCamera>();
        AddInstantiator<Host::LightProbeCamera>();

        AddInstantiator<Host::LightProbeKernelFilter>();
    }    

    struct TestStruct
    {
        JitterableFloat     rotateA;
        JitterableFloat     rotateB;
        JitterableFloat     scaleA;
        JitterableFloat     scaleB;
        JitterableFloat     vertScale;
        JitterableFloat     crustThickness;
        JitterableFlags     faceMask;

        int                 numIterations;
        int                 foldType;
        int                 primitiveType;

        struct
        {
            int maxSpecularIterations;
            int maxDiffuseIterations;
            float cutoffThreshold;
            float escapeThreshold;
            float rayIncrement;
            float failThreshold;
            float rayKickoff;

            bool clipCameraRays;
            int clipShape;
        }
        sdf;

        RenderObjectParams tracable;
        //BidirectionalTransform transform;

        bool doTakeSnapshot;
    };

    template<typename T, typename... Pack>
    __global__ inline void KernelCreateDeviceInstance2(T** newInstance, Pack... args)
    {
        assert(newInstance);
        assert(!*newInstance);

        T* newObj = new T;
    }

    template<typename ObjectType, typename... Pack>
    __host__ inline ObjectType* InstantiateOnDevice2(Pack... args)
    {
        ObjectType** cu_tempBuffer;
        IsOk(cudaMalloc((void***)&cu_tempBuffer, sizeof(ObjectType*)));
        IsOk(cudaMemset(cu_tempBuffer, 0, sizeof(ObjectType*)));

        KernelCreateDeviceInstance2 << <1, 1 >> > (cu_tempBuffer, args...);
        IsOk(cudaDeviceSynchronize());

        ObjectType* cu_data = nullptr;
        IsOk(cudaMemcpy(&cu_data, cu_tempBuffer, sizeof(ObjectType*), cudaMemcpyDeviceToHost));
        IsOk(cudaFree(cu_tempBuffer));

        Log::System("Instantiated device object at 0x%x\n", cu_data);
        return cu_data;
    }

    void Test()
    {
        auto cu_deviceData = InstantiateOnDevice2<TestStruct>();

        if(cu_deviceData) DestroyOnDevice(cu_deviceData);
    }         
   
    __host__ void RenderObjectFactory::InstantiateList(const ::Json::Node& node, const AssetType& expectedType, const std::string& objectTypeStr, AssetHandle<RenderObjectContainer>& renderObjects)
    {
        //Test();
        //return;
        
        for (::Json::Node::ConstIterator it = node.begin(); it != node.end(); ++it)
        {
            AssetHandle<Host::RenderObject> newObject;
            std::string newId = it.Name();
            ::Json::Node childNode = *it;

            if (newId.empty())
            {
                Log::Warning("Warning: skipping object with empty ID.\n");
                continue;
            }

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

                // Get the object instance flags
                auto& flagsFunctor = m_instanceFlagFunctors.find(newClass);
                Assert(flagsFunctor != m_instanceFlagFunctors.end());
                const uint instanceFlags = (flagsFunctor->second)();

                int numInstances = 1;
                if (childNode.GetValue("instances", numInstances, ::Json::kSilent))
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

                    newObject = (instantiator->second)(instanceId, expectedType, childNode);
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
                    newObject->SetHostStream(m_hostStream);
                    renderObjects->Emplace(newObject);

                    // The render object may have generated some of its own assets. Add them to the object list. 
                    std::vector<AssetHandle<Host::RenderObject>> childObjects = newObject->GetChildObjectHandles();
                    if (!childObjects.empty())
                    {
                        Log::Debug("Captured %i child objects:\n", childObjects.size());
                        Log::Indent indent2;
                        for (auto& child : childObjects)
                        {
                            AssertMsg(child, "A captured child object handle is invalid.");
                            child->SetHostStream(m_hostStream);
                            renderObjects->Emplace(child);

                            Log::Debug("%s\n", child->GetAssetID());
                        }
                    }
                }
            }
        }
    }

    __host__ void RenderObjectFactory::InstantiateSceneObjects(const ::Json::Node& rootNode, AssetHandle<RenderObjectContainer>& renderObjects)
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
    }
    
    __host__ void RenderObjectFactory::InstantiatePeripherals(const ::Json::Node& rootNode, AssetHandle<RenderObjectContainer>& renderObjects)
    {               
        Assert(renderObjects);

        {
            const ::Json::Node childNode = rootNode.GetChildObject("cameras", ::Json::kRequiredAssert);
            InstantiateList(childNode, AssetType::kCamera, "camera", renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("integrators", ::Json::kRequiredAssert);
            InstantiateList(childNode, AssetType::kIntegrator, "integrator", renderObjects);
        }
        {
            const ::Json::Node childNode = rootNode.GetChildObject("filters", ::Json::kRequiredAssert);
            InstantiateList(childNode, AssetType::kLightProbeFilter, "filter", renderObjects);
        }
    }
}