#pragma once

#include "UIAttribute.h"

namespace Enso
{  
    class UIGenericObject
    {
    public:
        UIGenericObject(const std::string& id, const SerialisableObjectSchema& schema);

        void Construct();

    private:
        std::string                                                     m_id;
        std::map<std::string, std::shared_ptr<UIGenericAttribute>>      m_attributeMap;
        std::vector<std::shared_ptr<UIGenericAttribute>>                m_attributeList;
    };

    class UIObjectContainer
    {
    public:
        UIObjectContainer();

        void Construct();
        
        void OnAddObject(const Json::Node& node);
        void OnDeleteObject(const Json::Node& node);
        void OnUpdateObject(const Json::Node& node);

    private:
        std::unordered_map<std::string, std::shared_ptr<UIGenericObject>>   m_objectMap;
    };
}