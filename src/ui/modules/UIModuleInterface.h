#pragma once

#include "io/json/JsonCommandQueue.h"
#include "win/D3DHeaders.h"

#include <unordered_map>

namespace Enso
{
    class UIGenericObject;
    
    class UIModuleInterface
    {
    public:
        UIModuleInterface(const std::string& id, Json::CommandQueue& commandQueue) : m_componentId(id), m_commandQueue(commandQueue) {}

        virtual void ConstructComponent() = 0;

    protected:
        const std::string   m_componentId;
        Json::CommandQueue& m_commandQueue;

        std::unordered_map<std::string, std::shared_ptr<UIGenericObject>>   m_uiObjectMap;
    };
}