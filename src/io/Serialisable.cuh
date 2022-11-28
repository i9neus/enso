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

    struct SerialisableAttributeProperties
    {
        __host__ SerialisableAttributeProperties();

        __host__ void Finalise();

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
        using AttributeList = std::vector<std::shared_ptr<SerialisableAttributeProperties>>;

    public:
        __host__ SerialisableObjectSchema(const std::string& id) : m_id(id) {}
        __host__ SerialisableObjectSchema(const std::string&, const Json::Node&);

        __host__ void Serialise(Json::Node& node) const;
        __host__ void Deserialise(const Json::Node& node);

        __host__ const std::string& GetId() const { return m_id; }
        __host__ const AttributeList GetAttributes() const { return m_attributes; }

        __host__ static const std::vector<std::string>& GetDataTypeStrings();
        __host__ static const std::vector<std::string>& GetWidgetTypeStrings();

    private:
        std::string         m_id;
        AttributeList       m_attributes;
    };

    class SerialisableObjectSchemaContainer
    {
    public:
        SerialisableObjectSchemaContainer() = delete;

        __host__ static std::shared_ptr<const SerialisableObjectSchema> FindSchema(const std::string&);
        __host__ static bool SchemaExists(const std::string&);

        __host__ static void Load(const std::string& jsonPath);
        __host__ static void RegisterSchema(const std::string& id, const SerialisableObjectSchema& schema);
        __host__ static void RegisterSchema(const std::string&, const std::string&);
        __host__ static void UnregisterSchema(const std::string& id);

    private:
        static std::unordered_map<std::string, std::shared_ptr<const SerialisableObjectSchema>> m_serialSchemaMap;
    };

    class Serialisable
    {
    public:
        __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const { return false; }
        __host__ virtual bool Deserialise(const Json::Node& rootNode, const int flags) { return false; }

    private:
    };
}