#include "UIAttribute.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

#include "io/json/JsonUtils.h"
#include "io/Serialisable.cuh"
#include "core/math/Math.cuh"

namespace Enso
{
    static void HelpMarker(const char* desc)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void UIGenericAttribute::Initialise(const SchemaAttributeProperties& properties, const Json::Node& node)
    {
        // Copy the schema data
        static_cast<SchemaAttributeProperties&>(*this) = properties;

        // Deserialise the data
        Deserialise(node);
    }

    bool UIGenericAttribute::Construct()
    {
        // Construct the full element
        m_isDirty = ConstructImpl();

        // Display the tooltip if specified
        if (!m_uiWidget.tooltip.empty())
        {
            HelpMarker(m_uiWidget.tooltip.c_str());
        }

        return m_isDirty;
    }
}