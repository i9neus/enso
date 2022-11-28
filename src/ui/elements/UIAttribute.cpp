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

    void UIGenericAttribute::Initialise(const SerialisableAttributeProperties& properties)
    {
        // Copy the properties
        static_cast<SerialisableAttributeProperties&>(*this) = properties;
    }

    bool UIGenericAttribute::Construct()
    {
        // Construct the full element
        if (!ConstructImpl()) { return false; }
        
        // Display the tooltip if specified
        if (!m_uiWidget.tooltip.empty())
        {
            HelpMarker(m_uiWidget.tooltip.c_str());
        }

        return true;
    }

#define UI_ATTRIBUTE_FLOAT_CONSTRUCT(Dimension, DataPtr) \
    bool UIAttributeFloat##Dimension::ConstructImpl() \
    {  \
        switch (m_uiWidget.type)  \
        {  \
        case kUIWidgetDrag:  \
            ImGui::DragFloat(m_uiWidget.label.c_str(), DataPtr, maxf(0.00001f, *DataPtr * 0.01f), m_dataRange[0], m_dataRange[1], "%.6f");  \
            break;  \
        case kUIWidgetSlider:  \
            ImGui::SliderFloat(m_uiWidget.label.c_str(), DataPtr, m_dataRange[0], m_dataRange[1], "%.6f");  \
            break;  \
        default:  \
            ImGui::InputFloat(m_uiWidget.label.c_str(), DataPtr, m_dataRange[0], m_dataRange[1], "%.6f");  \
            break;  \
        } \
        \
        return true; \
    }    

    UI_ATTRIBUTE_FLOAT_CONSTRUCT(, &m_data)
    UI_ATTRIBUTE_FLOAT_CONSTRUCT(2, &m_data[0])
    UI_ATTRIBUTE_FLOAT_CONSTRUCT(3, &m_data[0])
    UI_ATTRIBUTE_FLOAT_CONSTRUCT(4, &m_data[0])

#define UI_ATTRIBUTE_FLOAT_SERIALISE(Dimension, Serialiser) \
    void UIAttributeFloat##Dimension::Serialise(Json::Node& node) const \
    {  \
        node.##Serialiser(m_id, m_data); \
    }   

    UI_ATTRIBUTE_FLOAT_SERIALISE(, AddValue)
    UI_ATTRIBUTE_FLOAT_SERIALISE(2, AddVector)
    UI_ATTRIBUTE_FLOAT_SERIALISE(3, AddVector)
    UI_ATTRIBUTE_FLOAT_SERIALISE(4, AddVector)

#define UI_ATTRIBUTE_FLOAT_DESERIALISE(Dimension, Deserialiser) \
    void UIAttributeFloat##Dimension::Deserialise(const Json::Node& node) \
    {  \
        node.##Deserialiser(m_id, m_data, Json::kRequiredWarn); \
    }   

    UI_ATTRIBUTE_FLOAT_DESERIALISE(, GetValue)
    UI_ATTRIBUTE_FLOAT_DESERIALISE(2, GetVector)
    UI_ATTRIBUTE_FLOAT_DESERIALISE(3, GetVector)
    UI_ATTRIBUTE_FLOAT_DESERIALISE(4, GetVector)
}
