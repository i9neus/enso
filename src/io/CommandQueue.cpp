#include "CommandQueue.h"

namespace Enso
{
    CommandQueue::CommandQueue(const int maxQueueSize) :
        m_maxQueueSize(maxQueueSize)
    {}
    
    void CommandQueue::RegisterCommand(const std::string& commandId)
    {
        if (IsRegistered(commandId))
        {
            Log::Debug("RegisterCommand: command '%s' is already registered", commandId);
            return;
        }

        m_registeredCommands.emplace(commandId);
    }

    Json::Node CommandQueue::Create(const std::string& commandId)
    {
        AssertMsgFmt(m_maxQueueSize < 0 || m_queueSize < m_maxQueueSize, "Error: Exceeded maximum queue size.", m_maxQueueSize);
        
        AssertMsgFmt(IsRegistered(commandId), "Error: unknown command '%s'.", commandId.c_str());

        // Erase any existing command data and create a new object        
        m_commandStage.Clear();
        m_currentCommand = m_commandStage.AddChildObject(commandId);
        return m_currentCommand;
    }

    void CommandQueue::Enqueue()
    {
        AssertMsg(m_currentCommand, "Error: a command must be created before it can be enqueued.");

        std::lock_guard<std::mutex> lock(m_resourceMutex);        
        m_commandList.DeepCopy(m_commandStage);

        m_queueSize++;
        m_currentCommand = nullptr;
        m_commandStage.Clear();
    }

    bool CommandQueue::Flush(Json::Document& other)
    {
        std::lock_guard<std::mutex> lock(m_resourceMutex);

        if (m_commandList.NumMembers() == 0) { return false; }

        other.DeepCopy(m_commandList);
        m_commandList.Clear();
        m_queueSize = 0;
        return true;
    }

    void CommandQueue::Clear()
    {
        std::lock_guard<std::mutex> lock(m_resourceMutex);
        m_commandList.Clear();
    }

    std::string CommandQueue::Format()
    {
        std::lock_guard<std::mutex> lock(m_resourceMutex);        
        return (m_commandList.NumMembers() != 0) ? m_commandList.Stringify(true) : std::string("");
    }

    void CommandQueue::Echo()
    {
        const std::string formattedJson = Format();
        if(!formattedJson.empty())
        {
            Log::Debug(formattedJson);
        }
    }
}