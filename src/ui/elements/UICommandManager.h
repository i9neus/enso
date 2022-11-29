#pragma once

#include "UIGenericObject.h"

#include <unordered_map>
#include <functional>

namespace Enso
{
    class CommandQueue;
    
    class UICommandManager
    {
    public:
        UICommandManager(UIObjectContainer& container);
        
        void Flush(CommandQueue& cmdQueue);

        template<typename HostInstance, typename Lambda>
        void RegisterEventHandler(const std::string& eventId, HostInstance* host, Lambda functor)
        {
            m_eventMap[eventId] = std::bind(functor, host, std::placeholders::_1);
        }

    private:
        void OnCreateObject(const Json::Node&);
        void OnUpdateObject(const Json::Node&);
        void OnDeleteObject(const Json::Node&);

        bool ObjectExists(const std::string& objectId) const;        

    private:

        UIObjectContainer&                  m_objectContainer;
        std::shared_ptr<CommandQueue>       m_inboundCmdQueue;

        std::unordered_map<std::string, std::function<void(const Json::Node&)>>     m_eventMap;
    };
}