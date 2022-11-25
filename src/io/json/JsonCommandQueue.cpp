#include "JsonCommandQueue.h"

namespace Json
{
    CommandQueue::CommandQueue() = default;

    void CommandQueue::BeginComponent(const std::string& componentId)
    {
        Assert(!componentId.empty());
        m_componentId = componentId;
        m_componentJson = nullptr;
    }

    bool CommandQueue::BeginCommand(const std::string& commandId)
    {
        // Lazy initialisation of command node
        if (!m_componentJson)
        {
            AssertMsg(!m_componentId.empty(), "Command queue does not have a component set.");
            // TODO: Make JSON parser treat all IDs as literals rather than potential compound paths to save time.
            m_componentJson = m_rootDocument.GetChildObject(m_componentId, Json::kSilent | Json::kLiteralID);
            if (!m_componentJson)
            {
                m_componentJson = m_rootDocument.AddChildObject(m_componentId);
            }
        }

        // FIXME: Make new commands overwrite any pre-existing ones with the same name. Requires expanding Json class to allow objects to be erased, but this 
        // could be an issue with the rapidjson memory leaks.
        m_commandJson = m_componentJson.GetChildObject(commandId, Json::kSilent | Json::kLiteralID);
        if (m_commandJson) { return false; }

        m_commandJson = m_componentJson.AddChildObject(commandId);
        return true;
    }

    void CommandQueue::Push(const Json::Node& node)
    {
        AssertMsg(m_commandJson, "Command queue is not initialised.");
        m_commandJson.DeepCopy(node);
    }

    void CommandQueue::Clear()
    {
        m_rootDocument.Clear();
    }
}