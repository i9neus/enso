#pragma once

#include "core/math/Math.cuh"
#include "core/CudaHeaders.cuh"
#include "core/Assert.h"

#include <unordered_map>
#include <set>

namespace Enso
{
    namespace Json
    {
        class Node;
        class Document;
    }

    enum SerialDataType : int
    {
        kSerialDataUndefined = -1,

        kSerialDataBool,
        kSerialDataString,
        kSerialDataInt,
        kSerialDataInt2,
        kSerialDataInt3,
        kSerialDataInt4,
        kSerialDataFloat,
        kSerialDataFloat2,
        kSerialDataFloat3,
        kSerialDataFloat4,
        kSerialDataMat2,
        kSerialDataMat3,
        kSerialDataMat4,
    };

    enum SerialWidgetType : int
    {
        kUIWidgetUndefined = -1,
        kUIWidgetDefault = 0,

        kUIWidgetInput,
        kUIWidgetDrag,
        kUIWidgetSlider
    };

    struct SchemaAttributeProperties
    {
        SchemaAttributeProperties();

        void Finalise();

        std::string         m_id;
        int                 m_dataType;
        vec2                m_dataRange;

        struct
        {
            int                 type;
            std::string         label;
            std::string         tooltip;
        }
        m_uiWidget;

        std::set<std::string> isSet;
    };

    class SerialisableObjectSchema
    {
        using AttributeList = std::vector<std::shared_ptr<SchemaAttributeProperties>>;

    public:
        SerialisableObjectSchema(const std::string& id) : m_id(id) {}
        SerialisableObjectSchema(const std::string&, const Json::Node&);

        void Serialise(Json::Node& node) const;
        void Deserialise(const Json::Node& node);

        const std::string& GetId() const { return m_id; }
        const AttributeList GetAttributes() const { return m_attributes; }

        static const std::vector<std::string>& GetDataTypeStrings();
        static const std::vector<std::string>& GetWidgetTypeStrings();

    private:
        std::string         m_id;
        AttributeList       m_attributes;
    };

    class SerialisableObjectSchemaContainer
    {
    public:
        SerialisableObjectSchemaContainer() = delete;

        static std::shared_ptr<const SerialisableObjectSchema> FindSchema(const std::string&);
        static bool SchemaExists(const std::string&);

        static void Load(const std::string& jsonPath);
        static void RegisterSchema(const std::string& id, const SerialisableObjectSchema& schema);
        static void RegisterSchema(const std::string&, const std::string&);
        static void UnregisterSchema(const std::string& id);

    private:
        static std::unordered_map<std::string, std::shared_ptr<const SerialisableObjectSchema>> m_serialSchemaMap;
    };
}