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

    m_instantiators[Cuda::Host::QuadLight::GetAssetTypeString()] = QuadLightShelf::Instantiate;
    m_instantiators[Cuda::Host::SphereLight::GetAssetTypeString()] = SphereLightShelf::Instantiate;
    m_instantiators[Cuda::Host::EnvironmentLight::GetAssetTypeString()] = EnvironmentLightShelf::Instantiate;

    //m_instantiators[Cuda::Host::LambertBRDF::GetAssetTypeString()] = LambertBRDFShelf::Instantiate;

    m_instantiators[Cuda::Host::PerspectiveCamera::GetAssetTypeString()] = PerspectiveCameraShelf::Instantiate;
    m_instantiators[Cuda::Host::LightProbeCamera::GetAssetTypeString()] = LightProbeCameraShelf::Instantiate;
    m_instantiators[Cuda::Host::FisheyeCamera::GetAssetTypeString()] = FisheyeCameraShelf::Instantiate;

    m_instantiators[Cuda::Host::WavefrontTracer::GetAssetTypeString()] = WavefrontTracerShelf::Instantiate;
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
            Log::Debug("Skipped '%s': invalid DAG path.\n", object->GetAssetID().c_str());
            continue;
        }

        const std::string& dagPath = object->GetDAGPath();
        const Json::Node childNode = rootNode.GetChildObject(dagPath, Json::kSilent);

        AssertMsgFmt(childNode, "DAG path '%s' refers to missing or invalid JSON node.", dagPath.c_str());

        std::string objectClass;
        AssertMsgFmt(childNode.GetValue("class", objectClass, Json::kSilent),
            "Missing 'class' element in JSON object '%s'.", dagPath.c_str());

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

        newShelf->SetRenderObjectAttributes(object->GetAssetID(), dagPath, object->IsJitterable());
        newShelf->MakeClean();       

        shelves[object->GetDAGPath()] = newShelf;

        Log::Debug("Instantiated IMGUI shelf for '%s'.\n", dagPath);
    }

    return shelves;
}