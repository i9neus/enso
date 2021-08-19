#include "IMGUIAbstractShelf.h"
#include "generic/JsonUtils.h"

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

void IMGUIAbstractShelf::ConstructTransform(Cuda::BidirectionalTransform& transform, const bool isJitterable)
{
    if (ImGui::TreeNode("Transform"))
    {
        ImGui::PushID("p");
        ImGui::DragFloat3("Position", &transform.p.trans[0], math::max(0.01f, cwiseMax(transform.p.trans) * 0.01f));
        ImGui::DragFloat3("Rotation", &transform.p.rot[0], math::max(0.01f, cwiseMax(transform.p.rot) * 0.01f));
        ImGui::DragFloat("Scale", &transform.p.scale[0], math::max(0.01f, cwiseMax(transform.p.scale) * 0.01f));
        transform.p.scale = transform.p.scale[0];
        ImGui::PopID();

        if (isJitterable)
        {
            ImGui::PushID("dpdt");
            ImGui::DragFloat3("+/- Position", &transform.dpdt.trans[0], math::max(0.01f, cwiseMax(transform.dpdt.trans) * 0.01f));
            ImGui::DragFloat3("+/- Rotation", &transform.dpdt.rot[0], math::max(0.01f, cwiseMax(transform.dpdt.rot) * 0.01f));
            ImGui::DragFloat("+/- Scale", &transform.dpdt.scale[0], math::max(0.01f, cwiseMax(transform.dpdt.scale) * 0.01f));
            transform.dpdt.scale = transform.dpdt.scale[0];
            ImGui::PopID();

            ImGui::PushID("t");
            ImGui::SliderFloat3("~ Position", &transform.t.trans[0],0.0f, 1.0f);
            ImGui::SliderFloat3("~ Rotation", &transform.t.rot[0], 0.0f, 1.0f);
            ImGui::SliderFloat("~ Scale", &transform.t.scale[0], 0.0f, 1.0f);
            transform.t.scale = transform.t.scale[0];
            ImGui::PopID();
        }

        ImGui::TreePop();
    }
}

