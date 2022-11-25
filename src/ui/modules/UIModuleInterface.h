#pragma once

#include "io/json/JsonCommandQueue.h"

#include "win/D3DHeaders.h"

namespace Enso
{
    class UIModuleInterface
    {
    public:
        UIModuleInterface(const std::string& id, Json::CommandQueue& commandQueue) : m_componentId(id), m_commandQueue(commandQueue) {}

        virtual void ConstructComponent() = 0;

    protected:
        const std::string   m_componentId;
        Json::CommandQueue& m_commandQueue;
    };
}