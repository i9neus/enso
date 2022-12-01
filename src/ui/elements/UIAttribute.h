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
}
