#include "IMGUIElement.h"
#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"
#include "generic/StdIncludes.h"
#include "generic/JsonUtils.h"

#include "kernels/math/CudaMath.cuh"
#include "kernels/CudaHash.cuh"
#include "kernels/CudaAsset.cuh"

UIStyle::UIStyle(const int shelfIdx)
{
    /*const float alpha = 0.8f * shelfIdx++ / float(::max(1ull, m_shelves.size() - 1));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(alpha, 0.5f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(alpha, 0.6f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(alpha, 0.7f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(alpha, 0.9f, 0.9f));
    ImGui::PushStyleColor(ImGuiCol_Header, (ImVec4)ImColor::HSV(alpha, 0.9f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, (ImVec4)ImColor::HSV(alpha, 0.9f, 0.8f));*/

    const float lightness = Cuda::mix(0.5f, 0.35f, float(shelfIdx % 2));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.5f * lightness));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.5f * lightness));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.5f * lightness));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.9f * lightness));
    ImGui::PushStyleColor(ImGuiCol_Header, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.6f * lightness));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.6f * lightness));
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.6f * lightness));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.9f * lightness));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, (ImVec4)ImColor::HSV(0.0f, 0.0f, 1.0f));    
}

UIStyle::~UIStyle()
{
    ImGui::PopStyleColor(9);
}

IMGUIListBox::IMGUIListBox(const std::string& id, const std::string& addLabel, const std::string& overwriteLabel, 
                           const std::string& deleteLabel, const std::string& deleteAllLabel) :
    m_listBoxID(id),
    m_addLabel(addLabel),
    m_overwriteLabel(overwriteLabel),
    m_deleteLabel(deleteLabel),
    m_deleteAllLabel(deleteAllLabel),
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
    SL;
    //Erase a saved state from the container
    if (!m_deleteAllLabel.empty() && ImGui::Button(m_deleteAllLabel.c_str()))
    {
        if (!m_onDeleteAll || m_onDeleteAll())
        {
            Clear();
        }
    }

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

void IMGUIJitteredParameterTable::Push(const std::string& label, Cuda::JitterableFloat& param, const Cuda::vec3& range)
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
                    ImGui::PushItemWidth(100);
                    ImGui::DragFloat("+/-", &param.p, range.z * 0.01f, range.x, range.y, "%.6f"); SL;
                    ImGui::PopItemWidth();
                    ImGui::PushItemWidth(50);
                    ImGui::DragFloat("~", &param.dpdt, math::max(0.01f * range.z, param.dpdt * range.z), range.x, range.y, "%.6f"); SL;
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
    m_t(0.0f),
    m_switchLabels({ "Off", "On", "Rnd" })
{
    m_id = tfm::format("%s", id);    
}

void IMGUIJitteredFlagArray::Initialise(const std::vector<std::string>& flagLabels)
{
    Assert(!flagLabels.empty() && flagLabels.size() <= 32);
    m_flagLabels = flagLabels;
    m_numBits = m_flagLabels.size();
    m_states.resize(m_numBits);

    m_flags[0].resize(m_numBits);
    m_flags[1].resize(m_numBits);
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
    if (!ImGui::TreeNodeEx(m_id.c_str(), 0)) { return; }

    ImGui::PushID(m_pId.c_str());
    ImGui::PushItemWidth(30);
    for (int bit = 0; bit < m_numBits; ++bit)
    {        
        if (m_param.dpdt & (1 << bit)) { m_states[bit] = kFlagRnd; }
        else
        {
            m_states[bit] = (m_param.p >> bit) & 1;
        }

        if (ImGui::SliderInt(m_flagLabels[bit].c_str(), &m_states[bit], 0, 2, m_switchLabels[m_states[bit]].c_str()))
        {
            const uint mask = 1 << bit;
            switch(m_states[bit])
            {
            case kFlagOff:
                m_param.p &= ~mask;
                m_param.dpdt &= ~mask;
                break;
            case kFlagOn:
                m_param.p |= mask;
                m_param.dpdt &= ~mask;
                break;
            case kFlagRnd:
                m_param.p |= mask;
                m_param.dpdt |= mask;
                break;
            }
        }
        if (bit < m_numBits - 1) { ImGui::SameLine(); }
    }
    ImGui::PopItemWidth();
    ImGui::PopID();

    ImGui::TreePop();
}

void IMGUIElement::ConstructJitteredTransform(Cuda::BidirectionalTransform& transform, const bool isJitterable)
{
    if (!ImGui::TreeNodeEx("Transform", 0)) { return; }
 
    ImGui::PushID("pos");
    ImGui::DragFloat3("Position", &transform.trans.p[0], math::max(0.01f, cwiseMax(transform.trans.p) * 0.01f));
    if (isJitterable)
    {
        ImGui::DragFloat3("+/-", &transform.trans.dpdt[0], math::max(0.0001f, cwiseMax(transform.trans.dpdt) * 0.01f));
        ImGui::SliderFloat3("~", &transform.trans.t[0], 0.0f, 1.0f);
    }
    ImGui::PopID();

    ImGui::Separator();

    ImGui::PushID("rot");
    ImGui::DragFloat3("Rotation", &transform.rot.p[0], math::max(0.01f, cwiseMax(transform.rot.p) * 0.01f));
    if (isJitterable)
    {
        ImGui::DragFloat3("+/-", &transform.rot.dpdt[0], math::max(0.0001f, cwiseMax(transform.rot.dpdt) * 0.01f));
        ImGui::SliderFloat3("~", &transform.rot.t[0], 0.0f, 1.0f);
    }
    ImGui::PopID();

    ImGui::Separator();

    ImGui::PushID("sca");
    ImGui::DragFloat("Scale", &transform.scale.p[0], math::max(0.01f, cwiseMax(transform.scale.p) * 0.01f));
    if (isJitterable)
    {
        ImGui::DragFloat(" +/-", &transform.scale.dpdt[0], math::max(0.0001f, cwiseMax(transform.scale.dpdt) * 0.01f));
        ImGui::SliderFloat("~", &transform.scale.t[0], 0.0f, 1.0f);
    }
    ImGui::PopID();

    transform.scale.p = transform.scale.p[0];
    transform.scale.dpdt = transform.scale.dpdt[0];
    transform.scale.t = transform.scale.t[0];

    ImGui::TreePop(); 
} 



