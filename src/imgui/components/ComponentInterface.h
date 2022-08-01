#pragma once

#include "generic/JsonCommandQueue.h"

#include "generic/D3DIncludes.h"

namespace Gui
{
    class ComponentInterface
    {
    public:
        ComponentInterface(const std::string& id, Json::CommandQueue& commandQueue) : m_componentId(id), m_commandQueue(commandQueue) {}

        virtual void ConstructComponent() = 0;

    protected:
        const std::string   m_componentId;
        Json::CommandQueue&       m_commandQueue;
    };
}