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
            //case kSerialDataBool:     newAttr = std::make_shared<UIAttributeBool>(); break;
            //case kSerialDataString:   newAttr = std::make_shared<UIAttributeString>(); break;
            //case kSerialDataInt:      newAttr = std::make_shared<UIAttributeInt>(); break;
            //case kSerialDataInt2:     newAttr = std::make_shared<UIAttributeInt2>(); break;
            //case kSerialDataInt3:     newAttr = std::make_shared<UIAttributeInt3>(); break;
            //case kSerialDataInt4:     newAttr = std::make_shared<UIAttributeInt4>(); break;
            case kSerialDataFloat:      newAttr = std::make_shared<UIAttributeFloat>(); break;
            case kSerialDataFloat2:     newAttr = std::make_shared<UIAttributeFloat2>(); break;
            case kSerialDataFloat3:     newAttr = std::make_shared<UIAttributeFloat3>(); break;
            case kSerialDataFloat4:     newAttr = std::make_shared<UIAttributeFloat4>(); break;
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
        if (!ImGui::CollapsingHeader(m_id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return false; }
        
        // Construct the attributes in order
        bool isDirty = false;
        for (const auto& attribute : m_attributeList)
        {
            isDirty |= attribute->Construct();
        }

        return isDirty;
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
        for (const auto& attr : m_attributeList)
        {
            attr->Serialise(node);
        }
    }
}