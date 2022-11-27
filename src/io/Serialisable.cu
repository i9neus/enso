#include "Serialisable.cuh"
#include "json/JsonUtils.h"

namespace Enso
{
    SerialisableObjectSchemaContainer Serialisable::m_serialSchemaMap;
    
    __host__ SerialisableAttributeProperties::SerialisableAttributeProperties() : 
        dataType(kUIDataUndefined),
        dataRange(-kFltMax, kFltMax),
        widgetType(kUIWidgetUndefined)
    {

    }
    
    // Registers a schema defined in a JSON dictionary
    __host__ void Serialisable::RegisterSchema(const std::string& schemaId, const std::string& schema)
    {
        /*
            JSON schemas should be formatted as follows:
            {
                {
                    "id": "schemaId",
                    "dataType": "type",
                    "dataRange": [0.0, 1.0],
                    "widgetType": "type"
                },
                ....
            }
        */
        
        AssertMsgFmt(!FindSchema(schemaId), "Error: scheme was object '%s' is already registered.", schemaId);

        Json::Document json(schema);

        static const std::vector<std::string> dataTypeStr = { "kUIDataBool",
                                                              "kUIDataString",
                                                              "kUIDataInt", "kUIDataInt2", "kUIDataInt3",
                                                              "kUIDataFloat","kUIDataFloat2","kUIDataFloat3",
                                                              "kUIDataMat2", "kUIDataMat3", "kUIDataMat4" };

        static const std::vector<std::string> widgetTypeStr = { "kUIWidgetDefault", "kUIWidgetString", "kUIWidgetNumber", "kUIWidgetSlider" };

        SerialisableObjectSchema newSchema;
        newSchema.id = schemaId;

        // Load the schema attributes from the dictionary
        for (Json::Node::Iterator it = json.begin(); it != json.end(); ++it)
        {
            SerialisableAttributeProperties attr;
            auto node = *it;
            node.GetValue("id", attr.id, Json::kRequiredAssert | Json::kNotBlank);
            node.GetEnumeratedParameter("dataType", dataTypeStr, attr.dataType, Json::kRequiredAssert);
            node.GetVector("dataRange", attr.dataRange, Json::kSilent);
            node.GetEnumeratedParameter("widgetType", widgetTypeStr, attr.widgetType, Json::kSilent);

            newSchema.attributes.emplace(attr.id, attr);
        }

        RegisterSchema(schemaId, newSchema);
    }
    
    // Registers a pre-built schema
    __host__ void Serialisable::RegisterSchema(const std::string& schemaId, const SerialisableObjectSchema& schema)
    {
        AssertMsgFmt(!FindSchema(schemaId), "Error: scheme was object '%s' is already registered.", schemaId);

        m_serialSchemaMap.emplace(schemaId, std::make_shared<const SerialisableObjectSchema>(schema));
        Log::System("Registered schema for class '%s'", schemaId);
    }

    __host__ void Serialisable::UnregisterSchema(const std::string& schemaId)
    {
        auto it = m_serialSchemaMap.find(schemaId);
        AssertMsgFmt(it != m_serialSchemaMap.end(), "Error: schema for '%s' not found.", schemaId);

        m_serialSchemaMap.erase(it);
    }

    __host__ std::shared_ptr<const SerialisableObjectSchema> Serialisable::FindSchema(const std::string& schemaId)
    {
        auto it = m_serialSchemaMap.find(schemaId);
        return (it == m_serialSchemaMap.end()) ? std::shared_ptr<const SerialisableObjectSchema>(nullptr) : it->second;
    }
}