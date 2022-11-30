#pragma once

#include "CommandQueue.h"

namespace Enso
{
    class CommandManager
    {
    public:
        CommandManager();

        void Flush(CommandQueue& inCmd, const bool debug = false);

        template<typename HostInstance, typename Lambda>
        void RegisterEventHandler(const std::string& eventId, HostInstance* host, Lambda functor)
        {
            m_eventMap[eventId] = std::bind(functor, host, std::placeholders::_1);
        }

    private:
        std::shared_ptr<CommandQueue>                                               m_inboundCmdQueue;
        std::unordered_map<std::string, std::function<void(const Json::Node&)>>     m_eventMap;
    };
}