#include "WidgetInstantiators.h"
#include "generic/JsonUtils.h"

void SimpleMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(m_id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::ColorEdit3(tfm::format("Albedo (%s)", m_id).c_str(), (float*)&m_params[0].albedo);
    ImGui::ColorEdit3(tfm::format("Incandescence (%s)", m_id).c_str(), (float*)&m_params[0].incandescence);
}

void PlaneShelf::Construct()
{
    if (!ImGui::CollapsingHeader(m_id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::Checkbox("Bounded", &(m_params[0].isBounded));
}

void SphereShelf::Construct()
{
    if (!ImGui::CollapsingHeader(m_id.c_str())) { return; }

    ImGui::Text("[No attributes]");
}

void KIFSShelf::Construct()
{
    if (!ImGui::CollapsingHeader(m_id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }
    
    auto& p = m_params[0];
    ImGui::SliderFloat("Rotate X", &p.rotate.x, 0.0f, 1.0f);
    ImGui::SliderFloat("Rotate Y", &p.rotate.y, 0.0f, 1.0f);
    ImGui::SliderFloat("Rotate Z", &p.rotate.z, 0.0f, 1.0f);

    ImGui::SliderFloat("Scale A", &p.scale.x, 0.0f, 1.0f);
    ImGui::SliderFloat("Scale B", &p.scale.y, 0.0f, 1.0f);

    ImGui::SliderFloat("Crust thickness", &p.crustThickness, 0.0f, 1.0f);
    ImGui::SliderFloat("Vertex scale", &p.vertScale, 0.0f, 1.0f);

    ImGui::SliderInt("Iterations ", &p.numIterations, 0, kSDFMaxIterations);
}

void QuadLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(m_id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::InputFloat3("Position", &p.position[0]);
    ImGui::InputFloat3("Orientation", &p.orientation[0]);
    ImGui::InputFloat3("Scale", &p.scale[0]);

    ImGui::ColorEdit3("Colour", &p.colour[0]);
    ImGui::SliderFloat("Intensity", &p.intensity, -10.0f, 10.0f);
}

void EnvironmentLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(m_id.c_str())) { return; }

    ImGui::Text("[No attributes]");
}

void LambertBRDFShelf::Construct()
{
    if (!ImGui::CollapsingHeader(m_id.c_str())) { return; }

    ImGui::Text("[No attributes]");
}

void PerspectiveCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(m_id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::InputFloat3("Position", &p.position[0]);
    ImGui::InputFloat3("Look at", &p.lookAt[0]);

    ImGui::SliderFloat("F-stop", &p.fStop, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal length", &p.fLength, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal plane", &p.focalPlane, 0.0f, 2.0f);
}

void WavefrontTracerShelf::Construct()
{
    if (!ImGui::CollapsingHeader(m_id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::SliderInt("Max path depth", &p.maxDepth, 0, 20);
}

IMGUIShelfFactory::IMGUIShelfFactory()
{
    m_instantiators[Cuda::Host::SimpleMaterial::GetAssetTypeString()] = SimpleMaterialShelf::Instantiate;

    m_instantiators[Cuda::Host::Plane::GetAssetTypeString()] = PlaneShelf::Instantiate;
    m_instantiators[Cuda::Host::Sphere::GetAssetTypeString()] = SphereShelf::Instantiate;
    m_instantiators[Cuda::Host::KIFS::GetAssetTypeString()] = KIFSShelf::Instantiate;

    m_instantiators[Cuda::Host::QuadLight::GetAssetTypeString()] = QuadLightShelf::Instantiate;
    m_instantiators[Cuda::Host::EnvironmentLight::GetAssetTypeString()] = EnvironmentLightShelf::Instantiate;

    m_instantiators[Cuda::Host::LambertBRDF::GetAssetTypeString()] = LambertBRDFShelf::Instantiate;

    m_instantiators[Cuda::Host::PerspectiveCamera::GetAssetTypeString()] = PerspectiveCameraShelf::Instantiate;

    m_instantiators[Cuda::Host::WavefrontTracer::GetAssetTypeString()] = WavefrontTracerShelf::Instantiate;
}

std::vector<std::shared_ptr<IMGUIAbstractShelf>> IMGUIShelfFactory::Instantiate(const Json::Document& rootNode, const Cuda::RenderObjectContainer& renderObjects)
{
    Log::Indent indent("Setting up IMGUI shelves...\n");

    std::vector<std::shared_ptr<IMGUIAbstractShelf>> shelves;

    for (auto& object : renderObjects)
    {
        AssertMsgFmt(object->HasDAGPath(), "Object '%s' has an invalid DAG path.", object->GetAssetID().c_str());

        const std::string& dagPath = object->GetDAGPath();
        const Json::Node childNode = rootNode.GetChildObject(dagPath, Json::kSilent);

        AssertMsgFmt(childNode, "DAG path '%s' refers to missing or invalid JSON node.", dagPath.c_str());

        std::string objectClass;
        AssertMsgFmt(childNode.GetValue("class", objectClass, Json::kSilent),
            "Missing 'class' element in JSON object '%s'.", dagPath.c_str());

        auto& instantiator = m_instantiators.find(objectClass);
        if(instantiator == m_instantiators.end())
        {
            Log::Error("Error: '%s' is not a recognised object class.\n", objectClass);
            continue;
        }

        auto newShelf = (instantiator->second)(childNode);
        newShelf->SetIDAndDAGPath(object->GetAssetID(), dagPath);
        shelves.emplace_back(newShelf);

        Log::Debug("Instantiated IMGUI shelf for '%s'.\n", dagPath);
    }

    return shelves;
}