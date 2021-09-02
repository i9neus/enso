#include "IMGUIShelfFactory.h"

#include "IMGUIShelves.h"
#include "IMGUIKIFSShelf.h"

IMGUIShelfFactory::IMGUIShelfFactory()
{
    m_instantiators[Cuda::Host::SimpleMaterial::GetAssetTypeString()] = SimpleMaterialShelf::Instantiate;
    m_instantiators[Cuda::Host::CornellMaterial::GetAssetTypeString()] = CornellMaterialShelf::Instantiate;
    m_instantiators[Cuda::Host::KIFSMaterial::GetAssetTypeString()] = KIFSMaterialShelf::Instantiate;

    m_instantiators[Cuda::Host::Plane::GetAssetTypeString()] = PlaneShelf::Instantiate;
    m_instantiators[Cuda::Host::Sphere::GetAssetTypeString()] = SphereShelf::Instantiate;
    m_instantiators[Cuda::Host::KIFS::GetAssetTypeString()] = KIFSShelf::Instantiate;
    m_instantiators[Cuda::Host::CornellBox::GetAssetTypeString()] = CornellBoxShelf::Instantiate;
    m_instantiators[Cuda::Host::Box::GetAssetTypeString()] = BoxShelf::Instantiate;

    m_instantiators[Cuda::Host::QuadLight::GetAssetTypeString()] = QuadLightShelf::Instantiate;
    m_instantiators[Cuda::Host::SphereLight::GetAssetTypeString()] = SphereLightShelf::Instantiate;
    m_instantiators[Cuda::Host::EnvironmentLight::GetAssetTypeString()] = EnvironmentLightShelf::Instantiate;
    m_instantiators[Cuda::Host::DistantLight::GetAssetTypeString()] = DistantLightShelf::Instantiate;

    m_instantiators[Cuda::Host::LambertBRDF::GetAssetTypeString()] = LambertBRDFShelf::Instantiate;

    m_instantiators[Cuda::Host::PerspectiveCamera::GetAssetTypeString()] = PerspectiveCameraShelf::Instantiate;
    m_instantiators[Cuda::Host::LightProbeCamera::GetAssetTypeString()] = LightProbeCameraShelf::Instantiate;

    m_instantiators[Cuda::Host::WavefrontTracer::GetAssetTypeString()] = WavefrontTracerShelf::Instantiate;

    m_instantiators[Cuda::Host::LightProbeKernelFilter::GetAssetTypeString()] = LightProbeKernelFilterShelf::Instantiate;
}

std::map<std::string, std::shared_ptr<IMGUIAbstractShelf>> IMGUIShelfFactory::Instantiate(const Json::Document& rootNode, const Cuda::RenderObjectContainer& renderObjects)
{
    Log::Indent indent("Setting up IMGUI shelves...\n");

    std::map<std::string, std::shared_ptr<IMGUIAbstractShelf>> shelves;

    for (auto& object : renderObjects)
    {
        // Ignore objects instantiated by other objects
        if (object->IsChildObject()) { continue; }

        if (!object->HasDAGPath())
        {
            Log::Debug("Skipped '%s': missing DAG path.\n", object->GetAssetID().c_str());
            continue;
        }

        // Virtual DAG paths are appended with the instance index. Remove that to get the actual DAG path.
        const std::string& virtualDAGPath = object->GetDAGPath();
        std::string jsonDAGPath = virtualDAGPath;
        while (std::isdigit(jsonDAGPath.back())) { jsonDAGPath.pop_back(); }
        if (jsonDAGPath.empty() || jsonDAGPath.back() != Json::Node::kDAGDelimiter)
        {
            Log::Debug("Skipped '%s': invalid DAG path '%s'.\n", object->GetAssetID().c_str(), virtualDAGPath);
            continue;
        }
        jsonDAGPath.pop_back();

        const Json::Node childNode = rootNode.GetChildObject(jsonDAGPath, Json::kSilent);

        AssertMsgFmt(childNode, "DAG path '%s' refers to missing or invalid JSON node.", jsonDAGPath.c_str());

        std::string objectClass;
        AssertMsgFmt(childNode.GetValue("class", objectClass, Json::kSilent),
            "Missing 'class' element in JSON object '%s'.", jsonDAGPath.c_str());

        auto& instantiator = m_instantiators.find(objectClass);
        if (instantiator == m_instantiators.end())
        {
            Log::Debug("Skipped '%s': not a recognised object class.\n", objectClass);
            continue;
        }

        auto newShelf = (instantiator->second)(childNode);

        auto cast = object.DynamicCast<Cuda::Host::KIFS>();
        if (cast)
        {
            cast->GetAssetID();
        }

        newShelf->SetRenderObjectAttributes(object->GetAssetID(), virtualDAGPath);
        newShelf->MakeClean();       

        shelves[virtualDAGPath] = newShelf;

        Log::Debug("Instantiated IMGUI shelf for '%s'.\n", virtualDAGPath);
    }

    return shelves;
}