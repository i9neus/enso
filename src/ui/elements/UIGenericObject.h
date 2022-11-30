#pragma once

#include "UIAttribute.h"

namespace Enso
{  
    class UIGenericObject
    {
    public:
        UIGenericObject(const std::string& id, const SerialisableObjectSchema& schema, const Json::Node& node);

        bool Construct();
        bool IsDirty() const { return m_isDirty; }

        void Deserialise(const Json::Node&);
        void Serialise(Json::Node&) const;

    private:
        std::string                                                     m_id;
        std::map<std::string, std::shared_ptr<UIGenericAttribute>>      m_attributeMap;
        std::vector<std::shared_ptr<UIGenericAttribute>>                m_attributeList;

        bool                                                            m_isDirty;
    };

    using UIObjectContainer = std::unordered_map<std::string, std::shared_ptr<UIGenericObject>>;   
}