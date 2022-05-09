#include "IMGUIWidget.h"

#include "kernels/lightprobes/CudaLightProbeDataTransform.cuh"
#include "kernels/math/CudaMath.cuh"
#include "kernels/CudaAsset.cuh"

void IMGUIWidget::Text(const std::string& text, const ImColor colour)
{
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(colour));
    ImGui::Text(text.c_str());
    ImGui::PopStyleColor(1);
}

void IMGUIWidget::ConstructJitteredTransform(Cuda::BidirectionalTransform& transform, const bool isJitterable, const bool isNonlinearScale)
{
    if (!ImGui::TreeNodeEx("Transform", 0)) { return; }

    ImGui::PushID("pos");
    ImGui::DragFloat3("Position", &transform.trans.p[0], math::max(0.01f, cwiseMax(transform.trans.p) * 0.01f));
    HelpMarker("The base position of the transform.");

    if (isJitterable)
    {
        ImGui::DragFloat3("+/-", &transform.trans.dpdt[0], math::max(0.0001f, cwiseMax(transform.trans.dpdt) * 0.01f));
        HelpMarker("The relative range relative over which the transform position may be jittered.");

        ImGui::SliderFloat3("~", &transform.trans.t[0], 0.0f, 1.0f);
        HelpMarker("The mixing value that determines the evaluated position between + and - its range.");
    }
    ImGui::PopID();

    ImGui::Separator();

    ImGui::PushID("rot");
    ImGui::DragFloat3("Rotation", &transform.rot.p[0], 0.5f);
    HelpMarker("The base position of the rotation.");

    if (isJitterable)
    {
        ImGui::DragFloat3("+/-", &transform.rot.dpdt[0], 0.5f);
        HelpMarker("The relative range relative over which the transform rotation may be jittered.");

        ImGui::SliderFloat3("~", &transform.rot.t[0], 0.0f, 1.0f);
        HelpMarker("The mixing value that determines the evaluated rotation between + and - its range.");
    }
    ImGui::PopID();

    ImGui::Separator();

    ImGui::PushID("sca");
    if (isNonlinearScale)
    {
        ImGui::DragFloat3("Scale", &transform.scale.p[0], math::max(0.01f, cwiseMax(transform.scale.p) * 0.01f));
    }
    else
    {
        ImGui::DragFloat("Scale", &transform.scale.p[0], math::max(0.01f, cwiseMax(transform.scale.p) * 0.01f));
    }
    HelpMarker("The base scale of the transform.");

    if (isJitterable)
    {
        if (isNonlinearScale)
        {
            ImGui::DragFloat3(" +/-", &transform.scale.dpdt[0], math::max(0.0001f, cwiseMax(transform.scale.dpdt) * 0.01f));
        }
        else
        {
            ImGui::DragFloat(" +/-", &transform.scale.dpdt[0], math::max(0.0001f, cwiseMax(transform.scale.dpdt) * 0.01f));

        }
        HelpMarker("The relative range relative over which the transform scale may be jittered.");

        if (isNonlinearScale)
        {
            ImGui::SliderFloat3("~", &transform.scale.t[0], 0.0f, 1.0f);
        }
        else
        {
            ImGui::SliderFloat("~", &transform.scale.t[0], 0.0f, 1.0f);
        }
        HelpMarker("The mixing value that determines the evaluated scale between + and - its range.");
    }
    ImGui::PopID();

    if (isJitterable)
    {
        ImVec2 size = ImGui::GetItemRectSize();
        size.x = 75;
        int operation = -1;
        if (ImGui::Button("Shuffle", size)) { operation = Cuda::kJitterRandomise; } SL;
        if (ImGui::Button("Reset", size)) { operation = Cuda::kJitterReset; } SL;
        if (ImGui::Button("Flatten", size)) { operation = Cuda::kJitterFlatten; }

        if (operation != -1)
        {
            transform.trans.Update(operation);
            transform.rot.Update(operation);
            transform.scale.Update(operation);
        }
    }

    if (!isNonlinearScale)
    {
        transform.scale.p = transform.scale.p[0];
        transform.scale.dpdt = transform.scale.dpdt[0];
        transform.scale.t = transform.scale.t[0];
    }

    ImGui::TreePop();
}

void IMGUIWidget::ConstructFlagCheckBox(const std::string& name, const uint& mask, uint& flags)
{
    bool isEnabled = (flags & mask);
    ImGui::Checkbox(name.c_str(), &isEnabled);
    flags = (flags & ~mask) | ((uint(!isEnabled) - 1) & mask);
}

void IMGUIWidget::ConstructListBox(const std::string& id, const std::vector<std::string>& listItems, int& currentIdx)
{
    ImGui::PushID(id.c_str());

    if (ImGui::BeginListBox(id.c_str()))
    {
        for (int n = 0; n < listItems.size(); n++)
        {
            const bool isSelected = (currentIdx == n);
            if (ImGui::Selectable(listItems[n].c_str(), isSelected))
            {
                currentIdx = n;
            }
            if (isSelected) { ImGui::SetItemDefaultFocus(); }
        }
        ImGui::EndListBox();
    }

    ImGui::PopID();
}

void IMGUIWidget::ConstructComboBox(const std::string& name, const std::vector<std::string>& labels, int& selected)
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

void IMGUIWidget::ConstructProbeDataTransform(Cuda::LightProbeDataTransformParams& params)
{
    static const std::vector<std::string> kSwizzleLabels = { "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX" };
    
    ConstructComboBox("Position swizzle", kSwizzleLabels, params.posSwizzle);
    HelpMarker("The swizzle factor applied to the SH coefficients as they're baked out. Configure this value to match coordiante spaces between Unity and Probegen.");

    ImGui::Text("Invert position"); SL;
    ToolTip("Axis inverstion applied to the SH coefficients as they're baked out. Configure this value to match coordiante spaces between Unity and Probegen.");

    ImGui::PushID("posInvert");
    ImGui::Checkbox("X", &params.posInvertX); SL;
    ImGui::Checkbox("Y", &params.posInvertY); SL;
    ImGui::Checkbox("Z", &params.posInvertZ);
    ImGui::PopID();

    ConstructComboBox("SH swizzle", kSwizzleLabels, params.shSwizzle);
    HelpMarker("The swizzle factor applied to the SH coefficients as they're baked out. Configure this value to match coordiante spaces between Unity and Probegen.");

    ImGui::Text("Invert SH"); SL;
    ToolTip("Axis inverstion applied to the SH coefficients as they're baked out. Configure this value to match coordiante spaces between Unity and Probegen.");

    ImGui::PushID("shInvert");
    ImGui::Checkbox("X", &params.shInvertX); SL;
    ImGui::Checkbox("Y", &params.shInvertY); SL;
    ImGui::Checkbox("Z", &params.shInvertZ);
    ImGui::PopID();
}

void IMGUIJitteredColourPicker::Construct()
{
    ImGui::PushID(m_id.c_str());

    if (ImGui::BeginTable("", 2))
    {
        m_hsv[0] = m_param.p - m_param.dpdt;
        m_hsv[1] = m_param.p + m_param.dpdt;

        // We could also set ImGuiTableFlags_SizingFixedFit on the table and all columns will default to ImGuiTableColumnFlags_WidthFixed.
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 500.0f);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text(m_id.c_str());

        ImGui::TableSetColumnIndex(1);
        ImGui::ColorEdit3("->", &m_hsv[0][0], ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_InputHSV | ImGuiColorEditFlags_DisplayHSV); SL;
        ImGui::ColorEdit3("~", &m_hsv[1][0], ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_InputHSV | ImGuiColorEditFlags_DisplayHSV); SL;
        ImGui::PushItemWidth(150);
        ImGui::SliderFloat3("", &m_param.t[0], 0.0f, 1.0f);  SL;
        ImGui::PopItemWidth();

        m_param.p = mix(m_hsv[0], m_hsv[1], 0.5f);
        m_param.dpdt = (m_hsv[1] - m_hsv[0]) * 0.5f;

        if (ImGui::Button("s")) { m_param.Update(Cuda::kJitterRandomise); } ImGui::SameLine(0.0f, 1.0f);
        ToolTip("Shuffle");
        if (ImGui::Button("r")) { m_param.Update(Cuda::kJitterReset); } ImGui::SameLine(0.0f, 1.0f);
        ToolTip("Reset");
        if (ImGui::Button("f")) { m_param.Update(Cuda::kJitterFlatten); }
        ToolTip("Flatten");

        ImGui::EndTable();
    }

    ImGui::PopID();
}

void IMGUIJitteredColourPicker::Update()
{
    m_hsv[0] = m_param.p - m_param.dpdt;
    m_hsv[1] = m_param.p + m_param.dpdt;
}

void IMGUIJitteredParameterTable::Push(const std::string& label, const std::string& tooltip, Cuda::JitterableFloat& param, const Cuda::vec3& range)
{
    Assert(!label.empty());

    m_params.emplace_back(label, tooltip, param, range);
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
            auto& param = *(m_params[idx].param);
            const auto& range = m_params[idx].range;

            ImGui::TableNextRow();
            ImGui::PushID(m_params[idx].label.c_str());
            for (int column = 0; column < 2; column++)
            {
                ImGui::TableSetColumnIndex(column);
                if (column == 0)
                {
                    ImGui::Text(m_params[idx].label.c_str());
                    HelpMarker(m_params[idx].tooltip.c_str());
                }
                else
                {
                    ImGui::PushItemWidth(75);
                    ImGui::DragFloat("+/-", &param.p, range.z * 0.01f, range.x, range.y, "%.6f"); SL;
                    ImGui::PopItemWidth();
                    ImGui::PushItemWidth(50);
                    ImGui::DragFloat("~", &param.dpdt, math::max(0.01f * range.z, param.dpdt * range.z), range.x, range.y, "%.6f"); SL;
                    ImGui::SliderFloat("", &param.t, 0.0f, 1.0f); SL;
                    ImGui::PopItemWidth();

                    if (ImGui::Button("s")) { param.Update(Cuda::kJitterRandomise); } ImGui::SameLine(0.0f, 1.0f);
                    ToolTip("Shuffle");
                    if (ImGui::Button("r")) { param.Update(Cuda::kJitterReset); } ImGui::SameLine(0.0f, 1.0f);
                    ToolTip("Reset");
                    if (ImGui::Button("f")) { param.Update(Cuda::kJitterFlatten); }
                    ToolTip("Flatten");
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
            switch (m_states[bit])
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

IMGUIInputText::IMGUIInputText(const std::string& label, const std::string& contents, const std::string& id, const int minSize) :
    m_minSize(minSize),
    m_id(id),
    m_label(label)
{
    *this = contents;
}

IMGUIInputText& IMGUIInputText::operator=(const std::string& str)
{
    m_textData.resize(math::max(m_minSize, int(str.length())));
    std::memset(m_textData.data(), '\0', sizeof(char) * m_textData.size());
    std::memcpy(m_textData.data(), str.data(), sizeof(char) * str.length());

    return *this;
}

IMGUIInputText::operator std::string() const
{
    return std::string(m_textData.data());
}

void IMGUIInputText::Construct()
{
    if (m_id.empty())
    {
        ImGui::InputText(m_label.c_str(), m_textData.data(), m_textData.size());
    }
    else
    {
        ImGui::PushID(m_id.c_str());
        ImGui::InputText(m_label.c_str(), m_textData.data(), m_textData.size());
        ImGui::PopID();
    }
}