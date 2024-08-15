#include "SerialisableObjectSchema.h"

#include "json/JsonUtils.h"
#include "FileSystemUtils.h"
#include "core/utils/StdUtils.h"

namespace Enso
{
    /*
          Schemas attributes should be formatted as follows:
          {
              "attrId":
              {
                  "dataType": "type",
                  "dataRange": [0.0, 1.0],
                  "widgetType": "type"
              },
              ....
          }
      */

    std::unordered_map<std::string, std::shared_ptr<const SerialisableObjectSchema>> SerialisableObjectSchemaContainer::m_serialSchemaMap;

    SchemaAttributeProperties::SchemaAttributeProperties() :
        m_dataType(kSerialDataUndefined),
        m_dataRange(-kFltMax, kFltMax)
    {
        m_uiWidget.type = kUIWidgetUndefined;
    }

    void SchemaAttributeProperties::Finalise()
    {
        // If the labele is blank, set it to the ID
        if (m_uiWidget.label.empty()) { m_uiWidget.label = m_id; }
    }

    SerialisableObjectSchema::SerialisableObjectSchema(const std::string& schemaId, const Json::Node& node) :
        m_id(schemaId)
    {
        Deserialise(node);
    }

    const std::vector<std::string>& SerialisableObjectSchema::GetDataTypeStrings()
    {
        static const std::vector<std::string> dataTypeStr = { "bool",
                                                              "string",
                                                              "int", "int2", "int3", "int4",
                                                              "float","float2","float3", "float4",
                                                              "mat2", "mat3", "mat4" };
        return dataTypeStr;
    }

    const std::vector<std::string>& SerialisableObjectSchema::GetWidgetTypeStrings()
    {
        static const std::vector<std::string> widgetTypeStr = { "default", "input", "drag", "slider", "colourpicker" };
        return widgetTypeStr;
    }

    void SerialisableObjectSchema::Serialise(Json::Node& rootNode) const
    {
        Json::Node objectNode = rootNode.AddChildObject(m_id);
        Json::Node attrNode = objectNode.AddChildObject("attributes");

        for (const auto& it : m_attributes)
        {
            const auto& attr = *it;
            auto attrJson = attrNode.AddChildObject(attr.m_id);

            attrJson.AddEnumeratedParameter("dataType", GetDataTypeStrings(), attr.m_dataType);
            if (Contains(attr.isSet, "dataRange")) { attrJson.AddVector("dataRange", attr.m_dataRange); }

            if (Contains(attr.isSet, "widgetType")) { attrJson.AddEnumeratedParameter("widgetType", GetWidgetTypeStrings(), attr.m_uiWidget.type); }
            if (Contains(attr.isSet, "widgetLabel")) { attrJson.AddValue("widgetLabel", attr.m_uiWidget.label); }
            if (Contains(attr.isSet, "widgetTooltip")) { attrJson.AddValue("widgetTooltip", attr.m_uiWidget.tooltip); }
        }
    }

    void SerialisableObjectSchema::Deserialise(const Json::Node& node)
    {
        // Load the schema attributes from the dictionary
        Assert(node.IsObject());
        for (Json::Node::ConstIterator it = node.begin(); it != node.end(); ++it)
        {
            SchemaAttributeProperties attr;
            auto node = *it;
            attr.m_id = it.Name();
            AssertMsg(!attr.m_id.empty(), "Attribute ID cannot be blank.");

            node.GetEnumeratedParameter("dataType", GetDataTypeStrings(), attr.m_dataType, Json::kRequiredAssert);
            if (node.GetVector("dataRange", attr.m_dataRange, Json::kSilent)) { attr.isSet.insert("dataRange"); }
            if (node.GetValue("widgetLabel", attr.m_uiWidget.label, Json::kSilent)) { attr.isSet.insert("widgetLabel"); }
            if (node.GetEnumeratedParameter("widgetType", GetWidgetTypeStrings(), attr.m_uiWidget.type, Json::kSilent)) { attr.isSet.insert("widgetType"); }

            // Finalise and emplace the new attribute
            attr.Finalise();
            m_attributes.emplace_back(std::make_shared<SchemaAttributeProperties>(attr));
        }
    }

    void SerialisableObjectSchemaContainer::Load(const std::string& jsonPath)
    {
        const std::string jsonStr = ReadTextFile(jsonPath);
        Assert(!jsonStr.empty());

        const Json::Document document(jsonStr);

        Assert(document.IsObject());
        for (Json::Node::ConstIterator it = document.begin(); it != document.end(); ++it)
        {
            const auto& schemaId = it.Name();
            Assert(!schemaId.empty());
            AssertMsgFmt(!SchemaExists(schemaId), "Error: a schema with ID '%s' has already been registered.", schemaId.c_str());

            SerialisableObjectSchema newSchema(it.Name());
            newSchema.Deserialise(*it);

            // Register in the map
            RegisterSchema(schemaId, newSchema);
        }

        Log::System("Loaded %i object schemas from '%s'", m_serialSchemaMap.size(), jsonPath);
    }

    // Registers a schema defined by a raw JSON string
    void SerialisableObjectSchemaContainer::RegisterSchema(const std::string& schemaId, const std::string& schema)
    {
        AssertMsgFmt(!FindSchema(schemaId), "Error: schema was object '%s' is already registered.", schemaId);

        // Deserialise the schema from the JSON dictionary
        Json::Document json(schema);
        SerialisableObjectSchema newSchema(schemaId, json);

        // Register in the map
        RegisterSchema(schemaId, newSchema);
    }

    // Registers a pre-built schema
    void SerialisableObjectSchemaContainer::RegisterSchema(const std::string& schemaId, const SerialisableObjectSchema& schema)
    {
        AssertMsgFmt(!FindSchema(schemaId), "Error: schema was object '%s' is already registered.", schemaId);

        m_serialSchemaMap.emplace(schemaId, std::make_shared<const SerialisableObjectSchema>(schema));
        Log::System("Registered schema for class '%s'", schemaId);
    }

    // Unregister a schema
    void SerialisableObjectSchemaContainer::UnregisterSchema(const std::string& schemaId)
    {
        auto it = m_serialSchemaMap.find(schemaId);
        AssertMsgFmt(it != m_serialSchemaMap.end(), "Error: schema for '%s' not found.", schemaId);

        m_serialSchemaMap.erase(it);
    }

    // Finds and returns a smart pointer to a registered schema
    std::shared_ptr<const SerialisableObjectSchema> SerialisableObjectSchemaContainer::FindSchema(const std::string& schemaId)
    {
        auto it = m_serialSchemaMap.find(schemaId);
        return (it == m_serialSchemaMap.end()) ? std::shared_ptr<const SerialisableObjectSchema>(nullptr) : it->second;
    }

    // Checks to see whether a schema exists in the map
    bool SerialisableObjectSchemaContainer::SchemaExists(const std::string& schemaId)
    {
        return m_serialSchemaMap.find(schemaId) != m_serialSchemaMap.end();
    }
}