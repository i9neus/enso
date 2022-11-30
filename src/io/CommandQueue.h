#pragma once

#include "json/JsonUtils.h"

#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace Enso
{
    class CommandQueue
    {
        enum __attrs : int { kDefaultMaxQueueSize = 100 };

    public:
        using EventHandler = std::function<int(const Json::Node&)>;

        CommandQueue(const int maxQueueSize = kDefaultMaxQueueSize);
      
        void            RegisterCommand(const std::string& eventId);
        bool            IsRegistered(const std::string& commandId) const { return m_registeredCommands.find(commandId) != m_registeredCommands.end(); }       

        Json::Node      Create(const std::string& commandId);
        void            Enqueue();
        bool            Flush(Json::Document& other);
        void            Clear();

        int             Size() const { return m_queueSize; }
        inline bool     IsEmpty() { return m_queueSize == 0; }
        void            Echo();
        std::string     Format();

    private:
        std::unordered_map<std::string, EventHandler>   m_dispatchFunctors;
        std::unordered_set<std::string>                 m_registeredCommands;
        
        Json::Node          m_currentCommand;
        Json::Document      m_commandStage;
        Json::Document      m_commandList;

        std::mutex          m_resourceMutex;
        int                 m_maxQueueSize;
        std::atomic<int>    m_queueSize;
    };
}