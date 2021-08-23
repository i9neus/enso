#include "IMGUIElement.h"
#include "generic/JsonUtils.h"
#include "kernels/CudaHash.cuh"

IMGUIListBox::IMGUIListBox(const std::string& id, const std::string& addLabel, const std::string& overwriteLabel, const std::string& deleteLabel) :
    m_listBoxID(id),
    m_addLabel(addLabel),
    m_overwriteLabel(overwriteLabel),
    m_deleteLabel(deleteLabel),
    m_currentIdx(-1),
    m_lastIdx(-1)
{
    m_newItemIDData.resize(2048);
    std::memset(m_newItemIDData.data(), '\0', sizeof(char) * m_newItemIDData.size());
}

void IMGUIListBox::Construct()
{
    ImGui::PushID(m_listBoxID.c_str());

    if (ImGui::BeginListBox(m_listBoxID.c_str()))
    {
        auto it = m_listItems.begin();
        for (int n = 0; n < m_listItems.size(); n++, ++it)
        {
            const bool isSelected = (m_currentIdx == n);
            if (ImGui::Selectable(it->c_str(), isSelected))
            {
                m_currentIdx = n;
                if (m_onSelectItem && m_currentIdx != m_lastIdx)
                {
                    m_onSelectItem(*it);
                    m_lastIdx = m_currentIdx;
                }
            }
            if (isSelected) { ImGui::SetItemDefaultFocus(); }
        }
        ImGui::EndListBox(); 
    }

    // New element input control
    ImGui::InputText("", m_newItemIDData.data(), m_newItemIDData.size());

    // Save the current state to the container
    if (!m_addLabel.empty() && ImGui::Button(m_addLabel.c_str()))
    {
        std::string newElement = std::string(m_newItemIDData.data());
        if (!m_onAdd || m_onAdd(newElement))
        {
            Insert(newElement);
            std::memset(m_newItemIDData.data(), '\0', sizeof(char) * m_newItemIDData.size());
        }
    }
    SL;
    // Overwrite the currently selected state
    if (!m_overwriteLabel.empty() && ImGui::Button(m_overwriteLabel.c_str()) && m_currentIdx >= 0 && !m_listItems.empty())
    {
        std::list<std::string>::iterator it = m_listItems.begin();
        std::advance(it, m_currentIdx);
        if (it != m_listItems.end())
        {
            if (m_onOverwrite)
            {
                m_onOverwrite(*it);
            }
        }
    }
    SL;
    // Erase a saved state from the container
    if (!m_deleteLabel.empty() && ImGui::Button(m_deleteLabel.c_str()) && m_currentIdx >= 0 && !m_listItems.empty())
    {
        std::list<std::string>::iterator it = m_listItems.begin();
        std::advance(it, m_currentIdx);
        if (it != m_listItems.end())
        {
            if (!m_onDelete || m_onDelete(*it))
            {
                m_listItems.erase(it);
            }
        }
        m_currentIdx = -1;
    }
    /*SL;
    // Erase a saved state from the container
    if (ImGui::Button("Delete All"))
    {
        if (!m_onDeleteAll || m_onDeleteAll())
        {
            Clear();
        }
    }*/

    ImGui::PopID();
}

void IMGUIListBox::Clear()
{
    m_listItems.clear();
    m_currentIdx = -1;
}

void IMGUIListBox::Insert(const std::string& str)
{
    m_listItems.push_back(str);
    m_currentIdx = m_listItems.size() - 1;
}

std::string IMGUIListBox::GetCurrentlySelectedText() const
{
    if (m_currentIdx < 0) { return ""; }

    std::list<std::string>::const_iterator it = m_listItems.begin();
    std::advance(it, m_currentIdx);
    return (it != m_listItems.end()) ? *it : "";    
}

void IMGUIJitteredColourPicker::Construct()
{
    ImGui::PushID(m_id.c_str());

    m_hsv[0] = m_param.p - m_param.dpdt;
    m_hsv[1] = m_param.p + m_param.dpdt;

    if (ImGui::BeginTable("", 2))
    {
        // We could also set ImGuiTableFlags_SizingFixedFit on the table and all columns will default to ImGuiTableColumnFlags_WidthFixed.
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 500.0f);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text(m_id.c_str());

        ImGui::TableSetColumnIndex(1);
        ImGui::ColorEdit3("->", &m_hsv[0][0], ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_InputHSV | ImGuiColorEditFlags_DisplayHSV); SL;
        ImGui::ColorEdit3("~", &m_hsv[1][0], ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_InputHSV | ImGuiColorEditFlags_DisplayHSV); SL;
        ImGui::PushItemWidth(200);
        ImGui::SliderFloat3("", &m_param.t[0], 0.0f, 1.0f);
        ImGui::PopItemWidth();

        ImGui::EndTable();
    }

    m_param.p = mix(m_hsv[0], m_hsv[1], 0.5f);
    m_param.dpdt = abs(m_hsv[0] - m_hsv[1]) * 0.5f;

    ImGui::PopID();
}

void IMGUIJitteredColourPicker::Update()
{
    m_hsv[0] = m_param.p - m_param.dpdt;
    m_hsv[1] = m_param.p + m_param.dpdt;
}

void IMGUIElement::ConstructComboBox(const std::string& name, const std::vector<std::string>& labels, int& selected)
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

void IMGUIJitteredParameterTable::Push(const std::string& label, Cuda::JitterableFloat& param, const Cuda::vec2& range)
{
    Assert(!label.empty());

    m_params.emplace_back(label, param, range);
}

void IMGUIJitteredParameterTable::Construct()
{
    if (ImGui::BeginTable("", 2))
    {
        // We could also set ImGuiTableFlags_SizingFixedFit on the table and all columns will default to ImGuiTableColumnFlags_WidthFixed.
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 500.0f);

        for (int idx = 0; idx < m_params.size(); ++idx)
        {
            const auto& label = m_params[idx].label;
            auto& param = *(m_params[idx].param);
            const auto& range = m_params[idx].range;
            ImGui::TableNextRow();
            ImGui::PushID(label.c_str());
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
                    ImGui::DragFloat("+/-", &param.p, 0.001f, range.x, range.y, "%.6f"); SL;
                    ImGui::PopItemWidth();
                    ImGui::PushItemWidth(80);
                    ImGui::DragFloat("~", &param.dpdt, math::max(0.00001f, param.dpdt * 0.01f), 0.0f, 1.0f, "%.6f"); SL;
                    ImGui::SliderFloat("", &param.t, 0.0f, 1.0f);
                    ImGui::PopItemWidth();
                }
            }
            ImGui::PopID();
        }

        ImGui::EndTable();
    }
}

IMGUIJitteredFlagArray::IMGUIJitteredFlagArray(Cuda::JitterableFlags& param, const std::string& id) :
    m_param(param),
    m_t(0.0f)
{
    m_id = id;

    m_pId = tfm::format("%s p", m_id);
    m_dpdtId = tfm::format("%s dpdt", m_id);
    m_tId = tfm::format("%s t", m_id);
}

void IMGUIJitteredFlagArray::Initialise(const std::vector<std::string>& flagLabels)
{
    Assert(!flagLabels.empty() && flagLabels.size() <= 32);
    m_flagLabels = flagLabels;

    m_flags[0].resize(flagLabels.size());
    m_flags[1].resize(flagLabels.size());
}

void IMGUIJitteredFlagArray::Update()
{
    for (int bit = 0; bit < m_flags[0].size(); bit++)
    {
        m_flags[0][bit] = (m_param.p >> bit) & 1;
        m_flags[1][bit] = (m_param.dpdt >> bit) & 1;
    }
}

void IMGUIJitteredFlagArray::Construct()
{
    m_param.p = 0;
    m_param.dpdt = 0;
    
    // P flags
    ImGui::PushID(m_pId.c_str());
    for (int bit = 0; bit < m_flags[0].size(); ++bit)
    {
        bool value = m_flags[0][bit];
        ImGui::Checkbox(m_flagLabels[bit].c_str(), &value); SL;
        m_flags[0][bit] = value;

        m_param.p |= (1 << bit) * uint(value);
    }
    ImGui::Text(m_id.c_str());
    ImGui::PopID();

    // dPdT flags
    ImGui::PushID(m_dpdtId.c_str());
    for (int bit = 0; bit < m_flags[1].size(); ++bit)
    {
        bool value = m_flags[1][bit];
        ImGui::Checkbox(m_flagLabels[bit].c_str(), &value); SL;
        m_flags[1][bit] = value;

        m_param.dpdt |= (1 << bit) * uint(value);
    }
    ImGui::Text("+/-");
    ImGui::PopID();

    ImGui::PushID(m_tId.c_str());
    if (ImGui::SliderFloat("~", &m_t, 0.0f, 1.0f))
    {
        // For the t parameter, use a hash of the slider value rather than try and randomise each flag in turn
        const uint hash = Cuda::HashOf(uint(m_t * float(std::numeric_limits<uint>::max())));
        m_param.t = 0;
        for (int bit = 0; bit < m_flags[0].size(); ++bit)
        {
            m_param.t |= hash & (1 << bit);
        }
        Log::Error("Mask: %i\n", m_param.t);
    }
    ImGui::PopID();
}

void IMGUIElement::ConstructJitteredTransform(Cuda::BidirectionalTransform& transform, const bool isJitterable)
{
    if (ImGui::TreeNode("Transform"))
    {
        ImGui::PushID("p");
        ImGui::DragFloat3("Position", &transform.trans.p[0], math::max(0.01f, cwiseMax(transform.trans.p) * 0.01f));
        ImGui::DragFloat3("Rotation", &transform.rot.p[0], math::max(0.01f, cwiseMax(transform.rot.p) * 0.01f));
        ImGui::DragFloat("Scale", &transform.scale.p[0], math::max(0.01f, cwiseMax(transform.scale.p) * 0.01f));
        transform.scale.p = transform.scale.p[0];
        ImGui::PopID();

        if (isJitterable)
        {
            ImGui::PushID("dpdt");
            ImGui::DragFloat3("+/- Position", &transform.trans.dpdt[0], math::max(0.0001f, cwiseMax(transform.trans.dpdt) * 0.01f));
            ImGui::DragFloat3("+/- Rotation", &transform.rot.dpdt[0], math::max(0.0001f, cwiseMax(transform.rot.dpdt) * 0.01f));
            ImGui::DragFloat(" +/- Scale", &transform.scale.dpdt[0], math::max(0.0001f, cwiseMax(transform.scale.dpdt) * 0.01f));
            transform.scale.dpdt = transform.scale.dpdt[0];
            ImGui::PopID();

            ImGui::PushID("t");
            ImGui::SliderFloat3("~ Position", &transform.trans.t[0], 0.0f, 1.0f);
            ImGui::SliderFloat3("~ Rotation", &transform.rot.t[0], 0.0f, 1.0f);
            ImGui::SliderFloat("~ Scale", &transform.scale.t[0], 0.0f, 1.0f);
            transform.scale.t = transform.scale.t[0];
            ImGui::PopID();
        }

        ImGui::TreePop();
    }
} 



