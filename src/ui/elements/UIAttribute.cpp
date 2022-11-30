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

#define UI_ATTRIBUTE_FLOAT_CONSTRUCT(Dimension, DataPtr) \
    bool UIAttributeFloat##Dimension::ConstructImpl() \
    {  \
        switch (m_uiWidget.type)  \
        {  \
        case kUIWidgetDrag:  \
            return ImGui::DragFloat##Dimension(m_uiWidget.label.c_str(), DataPtr, maxf(0.00001f, *DataPtr * 0.01f), m_dataRange[0], m_dataRange[1], "%.6f");  \
        case kUIWidgetSlider:  \
            return ImGui::SliderFloat##Dimension(m_uiWidget.label.c_str(), DataPtr, m_dataRange[0], m_dataRange[1], "%.6f");  \
        default:  \
            return ImGui::InputFloat##Dimension(m_uiWidget.label.c_str(), DataPtr);  \
        } \
        return true; \
    }    

    UI_ATTRIBUTE_FLOAT_CONSTRUCT(, &m_data)
    UI_ATTRIBUTE_FLOAT_CONSTRUCT(2, &m_data[0])
    UI_ATTRIBUTE_FLOAT_CONSTRUCT(3, &m_data[0])
    UI_ATTRIBUTE_FLOAT_CONSTRUCT(4, &m_data[0])

#define UI_ATTRIBUTE_FLOAT_SERIALISE(Dimension, Serialiser) \
    void UIAttributeFloat##Dimension::Serialise(Json::Node& node) const \
    {  \
        node.##Serialiser(m_id, m_data, Json::kPathIsDAG); \
    }   

    UI_ATTRIBUTE_FLOAT_SERIALISE(, AddValue)
    UI_ATTRIBUTE_FLOAT_SERIALISE(2, AddVector)
    UI_ATTRIBUTE_FLOAT_SERIALISE(3, AddVector)
    UI_ATTRIBUTE_FLOAT_SERIALISE(4, AddVector)

#define UI_ATTRIBUTE_FLOAT_DESERIALISE(Dimension, Deserialiser) \
    void UIAttributeFloat##Dimension::Deserialise(const Json::Node& node) \
    {  \
        node.##Deserialiser(m_id, m_data, Json::kRequiredWarn | Json::kPathIsDAG); \
    }   

    UI_ATTRIBUTE_FLOAT_DESERIALISE(, GetValue)
    UI_ATTRIBUTE_FLOAT_DESERIALISE(2, GetVector)
    UI_ATTRIBUTE_FLOAT_DESERIALISE(3, GetVector)
    UI_ATTRIBUTE_FLOAT_DESERIALISE(4, GetVector)
}
