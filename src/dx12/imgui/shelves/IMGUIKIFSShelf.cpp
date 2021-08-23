#include "IMGUIKIFSShelf.h"
#include "generic/FilesystemUtils.h"

#include <random>

KIFSShelf::KIFSShelf(const Json::Node& json) : 
    IMGUIShelf(json),
    m_faceFlags(m_p.faceMask, "Faces"),
    m_jitteredParamTable("KIFS Params")
{
    m_faceFlags.Initialise(std::vector<std::string>({ "1", "2", "3", "4", "5", "6" }));
    m_jitteredParamTable.Push("Rotation A", m_p.rotateA, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Rotation B", m_p.rotateB, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Scale A", m_p.scaleA, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Scale B", m_p.scaleB, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Crust thickness", m_p.crustThickness, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Vertex scale", m_p.vertScale, Cuda::vec3(0.0f, 1.0f, 0.01f));
}

void KIFSShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructJitteredTransform(m_p.transform, true);

    m_jitteredParamTable.Construct();

    ImGui::SliderInt("Iterations ", &m_p.numIterations, 0, kSDFMaxIterations);
    ConstructComboBox("Fold type", std::vector<std::string>({ "Tetrahedron", "Cube" }), m_p.foldType);
    ConstructComboBox("Primitive type", std::vector<std::string>({ "Tetrahedron", "Cube" }), m_p.primitiveType);

    auto ConstructMaskCheckboxes = [](const std::string& label, uint& value, const int row) -> void
    {
        ImGui::PushID(row);
        for (int i = 0; i < 6; i++)
        {
            bool faceMaskBool = value & (1 << i);
            ImGui::Checkbox(tfm::format("%i", i).c_str(), &faceMaskBool); SL;
            value = (value & ~(1 << i)) | (int(faceMaskBool) << i);
        }
        ImGui::PopID();
        ImGui::Text(label.c_str());
    };

    m_faceFlags.Construct();

    ImGui::Checkbox("SDF Clip Camera Rays", &m_p.sdf.clipCameraRays);
    ConstructComboBox("SDF Clip Shape", std::vector<std::string>({ "Cube", "Sphere", "Torus" }), m_p.sdf.clipShape);
    ImGui::DragInt("SDF Max Specular Interations", &m_p.sdf.maxSpecularIterations, 1, 1, 500);
    ImGui::DragInt("SDF Max Diffuse Iterations", &m_p.sdf.maxDiffuseIterations, 1, 1, 500);
    ImGui::DragFloat("SDF Cutoff Threshold", &m_p.sdf.cutoffThreshold, math::max(0.00001f, m_p.sdf.cutoffThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
    ImGui::DragFloat("SDF Escape Threshold", &m_p.sdf.escapeThreshold, math::max(0.01f, m_p.sdf.escapeThreshold * 0.01f), 0.0f, 5.0f);
    ImGui::DragFloat("SDF Ray Increment", &m_p.sdf.rayIncrement, math::max(0.01f, m_p.sdf.rayIncrement * 0.01f), 0.0f, 2.0f);
    ImGui::DragFloat("SDF Ray Kickoff", &m_p.sdf.rayKickoff, math::max(0.01f, m_p.sdf.rayKickoff * 0.01f), 0.0f, 1.0f);
    ImGui::DragFloat("SDF Fail Threshold", &m_p.sdf.failThreshold, math::max(0.00001f, m_p.sdf.failThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
}

void KIFSShelf::Update()
{
    m_faceFlags.Update();
}

void KIFSShelf::Randomise(const Cuda::vec2 range)
{
    m_p.Randomise(range);
}

void KIFSShelf::JitterKIFSParameters()
{
    
}
