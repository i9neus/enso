#pragma once

#include "JsonUtils.h"

namespace Json
{
    class CommandQueue
    {
    public:
        CommandQueue();

        void BeginComponent(const std::string& componentId);
        bool BeginCommand(const std::string& commandId);

        template<typename T>
        void Push(const std::string& key, const T& value)
        {
            AssertMsg(m_commandJson, "Command queue is not initialised.");
            m_commandJson.AddValue(key, value);
        }
        void Push(const Json::Node& node);

        void Clear();

    private:
        Json::Document  m_rootDocument;
        Json::Node      m_componentJson;
        Json::Node      m_commandJson;

        std::string     m_componentId;
    };
}
