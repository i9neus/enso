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
        ImGui::DragFloat3("Position", &transform.jitterable.trans.p[0], math::max(0.01f, cwiseMax(transform.jitterable.trans.p) * 0.01f));
        ImGui::DragFloat3("Rotation", &transform.jitterable.rot.p[0], math::max(0.01f, cwiseMax(transform.jitterable.rot.p) * 0.01f));
        ImGui::DragFloat("Scale", &transform.jitterable.scale.p[0], math::max(0.01f, cwiseMax(transform.jitterable.scale.p) * 0.01f));
        transform.jitterable.scale.p = transform.jitterable.scale.p[0];
        ImGui::PopID();

        if (isJitterable)
        {
            ImGui::PushID("dpdt");
            ImGui::DragFloat3("+/- Position", &transform.jitterable.trans.dpdt[0], math::max(0.0001f, cwiseMax(transform.jitterable.trans.dpdt) * 0.01f));
            ImGui::DragFloat3("+/- Rotation", &transform.jitterable.rot.dpdt[0], math::max(0.0001f, cwiseMax(transform.jitterable.rot.dpdt) * 0.01f));
            ImGui::DragFloat(" +/- Scale", &transform.jitterable.scale.dpdt[0], math::max(0.0001f, cwiseMax(transform.jitterable.scale.dpdt) * 0.01f));
            transform.jitterable.scale.dpdt = transform.jitterable.scale.dpdt[0];
            ImGui::PopID();

            ImGui::PushID("t");
            ImGui::SliderFloat3("~ Position", &transform.jitterable.trans.t[0], 0.0f, 1.0f);
            ImGui::SliderFloat3("~ Rotation", &transform.jitterable.rot.t[0], 0.0f, 1.0f);
            ImGui::SliderFloat3("~ Scale", &transform.jitterable.scale.t[0], 0.0f, 1.0f);
            transform.jitterable.scale.t = transform.jitterable.scale.t[0];
            ImGui::PopID();
        }

        ImGui::TreePop();
    }
}

