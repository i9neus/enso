#include "UIGenericObject.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

namespace Enso
{
    UIGenericObject::UIGenericObject(const std::string& id, const SerialisableObjectSchema& schema) :
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
            newAttr->Initialise(*attribute);
            m_attributeMap.emplace(newAttr->m_id, newAttr);
            m_attributeList.emplace_back(newAttr);
        }
    }

    void UIGenericObject::Construct()
    {
        if (!ImGui::CollapsingHeader(m_id.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }
        
        // Construct the attributes in order
        for (const auto& attribute : m_attributeList)
        {
            attribute->Construct();
        }
    }
    
    UIObjectContainer::UIObjectContainer()
    {
    }

    void UIObjectContainer::Construct()
    {
        ImGui::Begin("Scene Objects");
        
        for (const auto& object : m_objectMap)
        {
            object.second->Construct();
        }

        ImGui::End();
    }

    void UIObjectContainer::OnAddObject(const Json::Node& node)
    {
        std::string objectId;
        node.GetValue("id", objectId, Json::kRequiredAssert | Json::kNotBlank);

        auto it = m_objectMap.find(objectId);
        AssertMsgFmt(it == m_objectMap.end(), "Error: UI object '%s' already exists in map.", objectId.c_str());

        std::string objectClass;
        node.GetValue("class", objectClass, Json::kRequiredAssert | Json::kNotBlank);

        auto schema = SerialisableObjectSchemaContainer::FindSchema(objectClass);
        AssertMsgFmt(schema, "Error: schema for class '%s' has not been registered.", objectClass.c_str());

        m_objectMap.emplace(objectId, std::make_shared<UIGenericObject>(objectId, *schema));
    }

    void UIObjectContainer::OnDeleteObject(const Json::Node& node)
    {

    }

    void UIObjectContainer::OnUpdateObject(const Json::Node& node)
    {

    }
}