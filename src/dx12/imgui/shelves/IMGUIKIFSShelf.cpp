#include "IMGUIKIFSShelf.h"
#include "generic/FilesystemUtils.h"

#include <random>

KIFSShelf::KIFSShelf(const Json::Node& json) : IMGUIShelf(json)
{
}

void KIFSShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.transform, true);

    const float TEXT_BASE_WIDTH = ImGui::CalcTextSize("A").x;

    auto ConstructRow = [](const std::string& label, Cuda::JitterableFloat& value, int row) -> void
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
                ImGui::DragFloat("+/-", &value.p, 0.001f, 0.0f, 1.0f, "%.6f"); SL;
                ImGui::PopItemWidth();
                ImGui::PushItemWidth(80);
                ImGui::DragFloat("~", &value.dpdt, math::max(0.00001f, value.dpdt * 0.01f), 0.0f, 1.0f, "%.6f"); SL;
                ImGui::SliderFloat("", &value.t, 0.0f, 1.0f);
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

    ConstructMaskCheckboxes("Face mask", p.faceMask.x, 0);
    ConstructMaskCheckboxes("Perturb", p.faceMask.y, 1);

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

void KIFSShelf::Randomise(const Cuda::vec2 range)
{
    m_params[0].Randomise(range);
}

void KIFSShelf::JitterKIFSParameters()
{
    
}
