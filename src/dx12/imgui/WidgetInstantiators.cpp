#include "WidgetInstantiators.h"
#include "generic/JsonUtils.h"

void SimpleMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::ColorEdit3(tfm::format("Albedo (%s)", m_id).c_str(), (float*)&p.albedo);
    ImGui::ColorEdit3(tfm::format("Incandescence (%s)", m_id).c_str(), (float*)&p.incandescence);
    ImGui::Checkbox("Use grid", &p.useGrid);
}

void CornellMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::ColorEdit3("Albedo 1", (float*)&p.albedo[0]);
    ImGui::ColorEdit3("Albedo 2", (float*)&p.albedo[1]);
    ImGui::ColorEdit3("Albedo 3", (float*)&p.albedo[2]);
    ImGui::ColorEdit3("Albedo 4", (float*)&p.albedo[3]);
    ImGui::ColorEdit3("Albedo 5", (float*)&p.albedo[4]);
    ImGui::ColorEdit3("Albedo 6", (float*)&p.albedo[5]);
}

void PlaneShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.tracable.transform);
    ImGui::Checkbox("Bounded", &p.isBounded);
}

void SphereShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.transform);
}

void CornellBoxShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.tracable.transform);
}

void KIFSShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.transform);

    ImGui::DragFloat2("Rotation", &p.rotate[0], math::max(0.01f, cwiseMax(p.rotate) * 0.01f));
    ImGui::DragFloat2("Scaling", &p.scale[0], math::max(0.01f, cwiseMax(p.scale) * 0.01f), 0.0f, 1.0f);
    ImGui::SliderFloat("Crust thickness", &p.crustThickness, 0.0f, 1.0f);
    ImGui::SliderFloat("Vertex scale", &p.vertScale, 0.0f, 1.0f);
    ImGui::SliderInt("Iterations ", &p.numIterations, 0, kSDFMaxIterations);

    for (int i = 0; i < 6; i++)
    {
        bool faceMaskBool = p.faceMask & (1 << i);
        ImGui::Checkbox(tfm::format("%i", i).c_str(), &faceMaskBool); 
        ImGui::SameLine();
        p.faceMask = (p.faceMask & ~(1 << i)) | (int(faceMaskBool) << i);

    }
    ImGui::Text("Face mask");

    ImGui::DragInt("SDF Max Specular Interations", &p.sdf.maxSpecularIterations, 1, 1, 500);
    ImGui::DragInt("SDF Max Diffuse Iterations", &p.sdf.maxDiffuseIterations, 1, 1, 500);
    ImGui::DragFloat("SDF Cutoff Threshold", &p.sdf.cutoffThreshold, math::max(0.00001f, p.sdf.cutoffThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
    ImGui::DragFloat("SDF Escape Threshold", &p.sdf.escapeThreshold, math::max(0.01f, p.sdf.escapeThreshold * 0.01f), 0.0f, 5.0f);
    ImGui::DragFloat("SDF Ray Increment", &p.sdf.rayIncrement, math::max(0.01f, p.sdf.rayIncrement * 0.01f), 0.0f, 2.0f);
    ImGui::DragFloat("SDF Fail Threshold", &p.sdf.failThreshold, math::max(0.00001f, p.sdf.failThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
}

void QuadLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.transform);

    ImGui::ColorEdit3("Colour", &p.colour[0]);
    ImGui::SliderFloat("Intensity", &p.intensity, -10.0f, 10.0f);
}

void EnvironmentLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str())) { return; }

    ImGui::Text("[No attributes]");
}

void LambertBRDFShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str())) { return; }

    ImGui::Text("[No attributes]");
}

void PerspectiveCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];

    ImGui::DragFloat3("Position", &p.position[0], math::max(0.01f, cwiseMax(p.position) * 0.01f));
    ImGui::DragFloat3("Look at", &p.lookAt[0], math::max(0.01f, cwiseMax(p.lookAt) * 0.01f));

    ImGui::SliderFloat("F-stop", &p.fStop, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal length", &p.fLength, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal plane", &p.focalPlane, 0.0f, 2.0f);
    ImGui::SliderFloat("Display gamma", &p.displayGamma, 0.01f, 5.0f);
}

void WavefrontTracerShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::SliderInt("Max path depth", &p.maxDepth, 0, 20);
    ImGui::ColorEdit3("Ambient radiance", &p.ambientRadiance[0]);
    ImGui::Checkbox("Debug normals", &p.debugNormals);
    ImGui::Checkbox("Debug shaders", &p.debugShaders);

    const char* labels[3] = { "MIS", "Lights", "BxDFs" };
    const char* selectedLabel = labels[p.importanceMode];  // Label to preview before opening the combo (technically it could be anything)
    if (ImGui::BeginCombo("Importance mode", selectedLabel, 0))
    {
        for (int n = 0; n < 3; n++)
        {
            const bool isSelected = (p.importanceMode == n);
            if (ImGui::Selectable(labels[n], isSelected))
            {
                p.importanceMode = n;
            }

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if (isSelected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

}

IMGUIShelfFactory::IMGUIShelfFactory()
{
    m_instantiators[Cuda::Host::SimpleMaterial::GetAssetTypeString()] = SimpleMaterialShelf::Instantiate;
    m_instantiators[Cuda::Host::CornellMaterial::GetAssetTypeString()] = CornellMaterialShelf::Instantiate;

    m_instantiators[Cuda::Host::Plane::GetAssetTypeString()] = PlaneShelf::Instantiate;
    m_instantiators[Cuda::Host::Sphere::GetAssetTypeString()] = SphereShelf::Instantiate;
    m_instantiators[Cuda::Host::KIFS::GetAssetTypeString()] = KIFSShelf::Instantiate;
    m_instantiators[Cuda::Host::CornellBox::GetAssetTypeString()] = CornellBoxShelf::Instantiate;

    m_instantiators[Cuda::Host::QuadLight::GetAssetTypeString()] = QuadLightShelf::Instantiate;
    m_instantiators[Cuda::Host::EnvironmentLight::GetAssetTypeString()] = EnvironmentLightShelf::Instantiate;

    //m_instantiators[Cuda::Host::LambertBRDF::GetAssetTypeString()] = LambertBRDFShelf::Instantiate;

    m_instantiators[Cuda::Host::PerspectiveCamera::GetAssetTypeString()] = PerspectiveCameraShelf::Instantiate;

    m_instantiators[Cuda::Host::WavefrontTracer::GetAssetTypeString()] = WavefrontTracerShelf::Instantiate;
}

std::vector<std::shared_ptr<IMGUIAbstractShelf>> IMGUIShelfFactory::Instantiate(const Json::Document& rootNode, const Cuda::RenderObjectContainer& renderObjects)
{
    Log::Indent indent("Setting up IMGUI shelves...\n");

    std::vector<std::shared_ptr<IMGUIAbstractShelf>> shelves;

    for (auto& object : renderObjects)
    {
        // Ignore objects instantiated by other objects
        if (object->IsChildObject()) { continue; }

        if (!object->HasDAGPath())
        {
            Log::Error("Error: object '%s' has an invalid DAG path.\n", object->GetAssetID().c_str());
            continue;
        }

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