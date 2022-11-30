#pragma once

#include "UIGenericObject.h"
#include "io/CommandManager.h"

#include <unordered_map>
#include <functional>

namespace Enso
{  
    class UICommandManager : public CommandManager
    {
    public:
        UICommandManager(UIObjectContainer& container);

    private:
        void OnCreateObject(const Json::Node&);
        void OnUpdateObject(const Json::Node&);
        void OnDeleteObject(const Json::Node&);

        bool ObjectExists(const std::string& objectId) const;        

    private:
        UIObjectContainer&                  m_objectContainer;
    };
}