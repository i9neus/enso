#include "UIGenericObject.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

namespace Enso
{
    UIGenericObject::UIGenericObject(const std::string& id, const SerialisableObjectSchema& schema, const Json::Node& node) :
        m_id(id)
    {
        for (auto& attribute : schema.GetAttributes())
        {
            std::shared_ptr<UIGenericAttribute> newAttr;
            switch (attribute->m_dataType)
            {
            case kSerialDataBool:     newAttr = std::make_shared<UIAttributeNumeric<bool, 1>>(); break;
            //case kSerialDataString:   newAttr = std::make_shared<UIAttributeString>(); break;
            case kSerialDataInt:        newAttr = std::make_shared<UIAttributeNumeric<int, 1>>(); break;
            case kSerialDataInt2:       newAttr = std::make_shared<UIAttributeNumeric<int, 2>>(); break;
            case kSerialDataInt3:       newAttr = std::make_shared<UIAttributeNumeric<int, 3>>(); break;
            case kSerialDataInt4:       newAttr = std::make_shared<UIAttributeNumeric<int, 4>>(); break;
            case kSerialDataFloat:      newAttr = std::make_shared<UIAttributeNumeric<float, 1>>(); break;
            case kSerialDataFloat2:     newAttr = std::make_shared<UIAttributeNumeric<float, 2>>(); break;
            case kSerialDataFloat3:     newAttr = std::make_shared<UIAttributeNumeric<float, 3>>(); break;
            case kSerialDataFloat4:     newAttr = std::make_shared<UIAttributeNumeric<float, 4>>(); break;
            //case kSerialDataMat2:     newAttr = std::make_shared<UIAttributeMat2>(); break;
            //case kSerialDataMat3:     newAttr = std::make_shared<UIAttributeMat3>(); break;
            //case kSerialDataMat4:     newAttr = std::make_shared<UIAttributeMat4>(); break;
            };

            // Initialise and emplace the new attribute
            newAttr->Initialise(*attribute, node);
            m_attributeMap.emplace(newAttr->m_id, newAttr);
            m_attributeList.emplace_back(newAttr);
        }
    }

    bool UIGenericObject::Construct()
    {
        m_isDirty = false;
        
        if (!ImGui::CollapsingHeader(m_id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return false; }

        ImGui::PushID(m_id.c_str());
        
        // Construct the attributes in order
        for (const auto& attribute : m_attributeList)
        {
            m_isDirty |= attribute->Construct();
        }

        ImGui::PopID();

        return m_isDirty;
    }

    void UIGenericObject::Deserialise(const Json::Node& node)
    {
        for (auto& attr : m_attributeList)
        {
            attr->Deserialise(node);
        }
    }

    void UIGenericObject::Serialise(Json::Node& node) const
    {
        Json::Node childNode = node.AddChildObject(m_id);
        for (const auto& attr : m_attributeList)
        {
            attr->Serialise(childNode);
        }
    }
}