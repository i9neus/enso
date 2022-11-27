#pragma once

#include "core/math/Math.cuh"
#include "core/CudaHeaders.cuh"
#include "core/Assert.h"

#include <unordered_map>

namespace Enso
{
    namespace Json
    {
        class Node;
        class Document;
    }

    enum SerialDataType : int
    {
        kUIDataUndefined = -1,

        kUIDataBool,
        kUIDataString,
        kUIDataInt,
        kUIDataInt2,
        kUIDataInt3,
        kUIDataFloat,
        kUIDataFloat2,
        kUIDataFloat3,
        kUIDataMat2,
        kUIDataMat3,
        kUIDataMat4,
    };

    enum SerialWidgetType : int
    {
        kUIWidgetUndefined = -1,
        kUIWidgetDefault = 0,

        kUIWidgetString,
        kUIWidgetNumber,
        kUIWidgetSlider
    };

    struct SerialisableAttributeProperties
    {
        __host__ SerialisableAttributeProperties();

        std::string         id;
        int                 dataType;
        vec2                dataRange;
        int                 widgetType;
    };

    struct SerialisableObjectSchema
    {
        std::string         id;
        std::unordered_map<std::string, SerialisableAttributeProperties> attributes;
    };

    using SerialisableObjectSchemaContainer = std::unordered_map<std::string, std::shared_ptr<const SerialisableObjectSchema>>;

    class Serialisable
    {
    public:
        __host__ virtual bool SerialiseSchema(Json::Node& rootNode, const int flags) { return false; }

        __host__ virtual bool SerialiseData(Json::Node& rootNode, const int flags) { return false; }
        __host__ virtual bool DeserialiseData(const Json::Node& rootNode, const int flags) { return false; }

    protected:
        __host__ Serialisable() {}

        __host__ static void RegisterSchema(const std::string& id, const SerialisableObjectSchema& schema);
        __host__ static void RegisterSchema(const std::string&, const std::string&);
        __host__ static void UnregisterSchema(const std::string& id);
        __host__ static std::shared_ptr<const SerialisableObjectSchema> FindSchema(const std::string&);

    private:
        static SerialisableObjectSchemaContainer m_serialSchemaMap;
    };

}