#pragma once

#include "core/math/Math.cuh"
#include "io/json/JsonUtils.h"
#include "io/SerialisableObjectSchema.h"

#include <functional>

namespace Enso
{
    namespace Json { class Node; }  

    class UIGenericAttribute : public SchemaAttributeProperties
    {
    public:
        UIGenericAttribute() : m_isDirty(false) {}
        ~UIGenericAttribute() = default;

        void Initialise(const SchemaAttributeProperties& properties, const Json::Node& node);
        virtual void Serialise(Json::Node&) const = 0;
        virtual void Deserialise(const Json::Node&) = 0;

        bool IsDirty() const { return m_isDirty; }
        void MakeClean() { m_isDirty = false; }

        bool Construct();

    protected:
        virtual bool ConstructImpl() = 0;

        bool            m_isDirty;
    };

#define UI_ATTRIBUTE_FLOAT(Dimension, DataType) \
    class UIAttributeFloat##Dimension : public UIGenericAttribute \
    { \
    public: \
        UIAttributeFloat##Dimension() : m_data(0.0f) {} \
        virtual void Serialise(Json::Node&) const override final; \
        virtual void Deserialise(const Json::Node&) override final; \
    \
    protected: \
        virtual bool ConstructImpl() override final; \
    \
    private: \
        DataType m_data; \
    }

    UI_ATTRIBUTE_FLOAT(, float);
    UI_ATTRIBUTE_FLOAT(2, vec2);
    UI_ATTRIBUTE_FLOAT(3, vec3);
    UI_ATTRIBUTE_FLOAT(4, vec4);

}
