#include "WidgetInstantiators.h"
#include "generic/JsonUtils.h"

#define SL ImGui::SameLine()

void IMGUIAbstractShelf::ConstructComboBox(const std::string& name, const std::vector<std::string>& labels, int& selected)
{
    std::string badLabel = "[INVALID VALUE]";
    const char* selectedLabel = (selected < 0 || selected >= labels.size()) ? badLabel.c_str() : labels[selected].c_str(); 

    if (ImGui::BeginCombo(name.c_str(), selectedLabel, 0))
    {
        for (int n = 0; n < labels.size(); n++)
        {
            const bool isSelected = (selected == n);
            if (ImGui::Selectable(labels[n].c_str(), isSelected))
            {
                selected = n;
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

void IMGUIAbstractShelf::ConstructTransform(Cuda::BidirectionalTransform& transform)
{
    if (ImGui::TreeNode("Transform"))
    {
        ImGui::DragFloat3("Position", &transform.trans[0], math::max(0.01f, cwiseMax(transform.trans) * 0.01f));
        ImGui::DragFloat3("Rotation", &transform.rot[0], math::max(0.01f, cwiseMax(transform.rot) * 0.01f));
        //ImGui::DragFloat3("Scale XYZ", &transform.scale[0], math::max(0.01f, cwiseMax(transform.scale) * 0.01f));
        ImGui::DragFloat("Scale XYZ", &transform.scale[0], math::max(0.01f, cwiseMax(transform.scale) * 0.01f));
        transform.scale = transform.scale[0];
        ImGui::TreePop();
    }
}

void SimpleMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::ColorEdit3(tfm::format("Albedo (%s)", m_id).c_str(), (float*)&p.albedo);
    ImGui::ColorEdit3(tfm::format("Incandescence (%s)", m_id).c_str(), (float*)&p.incandescence);
    ImGui::Checkbox("Use grid", &p.useGrid);
}

void KIFSMaterialShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::SliderFloat3("HSL lower", &p.hslLower[0], 0.0f, 1.0f);
    ImGui::SliderFloat3("HSL upper", &p.hslUpper[0], 0.0f, 1.0f);
    ImGui::ColorEdit3(tfm::format("Incandescence (%s)", m_id).c_str(), (float*)&p.incandescence);    
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

    ImGui::Checkbox("Exclude from bake", &p.excludeFromBake);
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

    const float TEXT_BASE_WIDTH = ImGui::CalcTextSize("A").x;

    auto ConstructRow = [](const std::string& label, Cuda::vec3& value, int row) -> void
    {
        ImGui::TableNextRow();
        ImGui::PushID(row);
        for (int column = 0; column < 2; column++)
        {
            ImGui::TableSetColumnIndex(column);
            if (column == 0)
            {
                ImGui::Text(label.c_str());
            }
            else
            {
                ImGui::PushItemWidth(140);
                ImGui::DragFloat("+/-", &value[0], math::max(0.01f, value[0] * 0.01f), 0.0f, 1.0f); SL;
                ImGui::PopItemWidth();
                ImGui::PushItemWidth(80);
                ImGui::DragFloat("~", &value[1], math::max(0.01f, value[1] * 0.01f), 0.0f, 1.0f); SL;
                ImGui::SliderFloat("", &value[2], 0.0f, 1.0f);
                ImGui::PopItemWidth();
            }
        }
        ImGui::PopID();
    };

    if (ImGui::BeginTable("", 2))
    {
        // We could also set ImGuiTableFlags_SizingFixedFit on the table and all columns will default to ImGuiTableColumnFlags_WidthFixed.
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 500.0f);       

        ConstructRow("Rotation A", p.rotateA, 0);
        ConstructRow("Rotation B", p.rotateB, 1);
        ConstructRow("Scale A", p.scaleA, 2);
        ConstructRow("Scale B", p.scaleB, 3);
        ConstructRow("Crust thickness", p.crustThickness, 4);
        ConstructRow("Vertex scale", p.vertScale, 5);
        
        ImGui::EndTable();
    }

    ImGui::SliderInt("Iterations ", &p.numIterations, 0, kSDFMaxIterations);
    ConstructComboBox("Fold type", std::vector<std::string>({ "Tetrahedron", "Cube" }), p.foldType);
    ConstructComboBox("Primitive type", std::vector<std::string>({ "Tetrahedron", "Cube" }), p.primitiveType);

    if (ImGui::Button("Snapshot")) { p.doTakeSnapshot = true; }

    for (int i = 0; i < 6; i++)
    {
        bool faceMaskBool = p.faceMask & (1 << i);
        ImGui::Checkbox(tfm::format("%i", i).c_str(), &faceMaskBool); SL;
        p.faceMask = (p.faceMask & ~(1 << i)) | (int(faceMaskBool) << i);

    }
    ImGui::Text("Face mask");

    ImGui::Checkbox("SDF Clip Camera Rays", &p.sdf.clipCameraRays);
    ConstructComboBox("SDF Clip Shape", std::vector<std::string>({ "Cube", "Sphere", "Torus" }), p.sdf.clipShape);
    ImGui::DragInt("SDF Max Specular Interations", &p.sdf.maxSpecularIterations, 1, 1, 500);
    ImGui::DragInt("SDF Max Diffuse Iterations", &p.sdf.maxDiffuseIterations, 1, 1, 500);
    ImGui::DragFloat("SDF Cutoff Threshold", &p.sdf.cutoffThreshold, math::max(0.00001f, p.sdf.cutoffThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
    ImGui::DragFloat("SDF Escape Threshold", &p.sdf.escapeThreshold, math::max(0.01f, p.sdf.escapeThreshold * 0.01f), 0.0f, 5.0f);
    ImGui::DragFloat("SDF Ray Increment", &p.sdf.rayIncrement, math::max(0.01f, p.sdf.rayIncrement * 0.01f), 0.0f, 2.0f);
    ImGui::DragFloat("SDF Ray Kickoff", &p.sdf.rayKickoff, math::max(0.01f, p.sdf.rayKickoff * 0.01f), 0.0f, 1.0f);
    ImGui::DragFloat("SDF Fail Threshold", &p.sdf.failThreshold, math::max(0.00001f, p.sdf.failThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
}

void KIFSShelf::Reset() 
{
}

void QuadLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.transform);

    ImGui::ColorEdit3("Colour", &p.colour[0]);
    ImGui::SliderFloat("Intensity", &p.intensity, -10.0f, 10.0f);
}

void SphereLightShelf::Construct()
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

    ImGui::Checkbox("Active", &p.camera.isActive); SL;
    ImGui::Checkbox("Live", &p.camera.isLive);  SL;
    ImGui::Checkbox("Realtime", &p.isRealtime);
    ImGui::Checkbox("Mimic light probe", &p.mimicLightProbe);
    
    ImGui::DragFloat3("Position", &p.position[0], math::max(0.01f, cwiseMax(p.position) * 0.01f));
    ImGui::DragFloat3("Look at", &p.lookAt[0], math::max(0.01f, cwiseMax(p.lookAt) * 0.01f));

    ImGui::SliderFloat("F-stop", &p.fStop, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal length", &p.fLength, 0.0f, 1.0f);
    ImGui::SliderFloat("Focal plane", &p.focalPlane, 0.0f, 2.0f);
    ImGui::SliderFloat("Display gamma", &p.displayGamma, 0.01f, 5.0f);

    ImGui::SliderInt("Max path depth", &p.camera.overrides.maxDepth, -1, 20);
    ImGui::DragFloat("Splat clamp", &p.camera.splatClamp, math::max(0.01f, p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());
}

LightProbeCameraShelf::LightProbeCameraShelf(const Json::Node& json)
    : IMGUIShelf(json)
{
    m_swizzleLabels = { "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX" };  
    m_pathData.resize(2048);
    m_pathData[0] = '\0';
}

void LightProbeCameraParamsUI::ToJson(Json::Node& node) const
{
    LightProbeCameraParams::ToJson(node);
    node.AddValue("usdExportPath", *usdExportPath);
}

void LightProbeCameraParamsUI::FromJson(const Json::Node& node, const int flags)
{
    LightProbeCameraParams::FromJson(node, flags);
    node.GetValue("usdExportPath", *usdExportPath, flags);
}

void LightProbeCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];

    ImGui::Checkbox("Active", &p.camera.isActive); SL;
    ImGui::Checkbox("Live", &p.camera.isLive);

    ConstructTransform(p.grid.transform);

    ImGui::InputInt3("Grid density", &p.grid.gridDensity[0]);
    ConstructComboBox("SH order", {"L0", "L1", "L2"}, p.grid.shOrder);    
    ImGui::SliderInt("Max path depth", &p.camera.overrides.maxDepth, -1, 20);
    ImGui::DragFloat("Splat clamp", &p.camera.splatClamp, math::max(0.01f, p.camera.splatClamp * 0.01f), 0.0f, std::numeric_limits<float>::max());

    ImGui::DragInt("Max samples", &p.maxSamples);

    ImGui::Checkbox("Debug PRef", &p.grid.debugOutputPRef); SL;
    ImGui::Checkbox("Debug validity", &p.grid.debugOutputValidity); SL;
    ImGui::Checkbox("Debug bake", &p.grid.debugBakePRef);

    if (ImGui::Button("Export")) { p.doExport = true; }
    
    // FIXME: This is incredibly ugly. Fix it asap.
    m_params[0].usdExportPath = m_params[1].usdExportPath = &m_usdExportPath[1];
    ImGui::InputText("USD path", m_pathData.data(), m_pathData.size(), ImGuiInputTextFlags_EnterReturnsTrue);
    {
        m_usdExportPath[1] = m_pathData.data();
        if (m_usdExportPath[1] != m_usdExportPath[0])
        {
            p.hasPathChanged = true;
        }
        m_usdExportPath[0] = m_usdExportPath[1];
    }

    ConstructComboBox("Swizzle", m_swizzleLabels, p.grid.axisSwizzle);    
    ImGui::Text("Invert axes"); SL;
    ImGui::Checkbox("X", &p.grid.invertX); SL;
    ImGui::Checkbox("Y", &p.grid.invertY); SL;
    ImGui::Checkbox("Z", &p.grid.invertZ);    
}

void LightProbeCameraShelf::Reset()
{
    m_params[0].doExport = m_params[1].doExport = false;
    m_params[0].hasPathChanged = m_params[1].hasPathChanged = false;
}

void FisheyeCameraShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];

    ImGui::Checkbox("Active", &p.camera.isActive); SL;
    ImGui::Checkbox("Live", &p.camera.isLive);

    ConstructTransform(p.transform);

    ImGui::SliderInt("Override max path depth", &p.camera.overrides.maxDepth, -1, 20);
}

void WavefrontTracerShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ImGui::SliderInt("Max path depth", &p.maxDepth, 0, 20);
    ImGui::ColorEdit3("Ambient radiance", &p.ambientRadiance[0]);
    ImGui::Checkbox("Debug normals", &p.debugNormals);
    ImGui::Checkbox("Debug shaders", &p.debugShaders);
    ConstructComboBox("Importance mode", { "MIS", "Lights", "BxDFs" }, p.importanceMode);
    ConstructComboBox("Trace mode", { "Wavefront", "Path" }, p.traceMode);
}

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
        if(instantiator == m_instantiators.end())
        {
            Log::Debug("Skipped '%s': not a recognised object class.\n", objectClass);
            continue;
        }

        auto newShelf = (instantiator->second)(childNode);
        newShelf->SetIDAndDAGPath(object->GetAssetID(), dagPath);
        shelves.emplace_back(newShelf);

        Log::Debug("Instantiated IMGUI shelf for '%s'.\n", dagPath);
    }

    return shelves;
}