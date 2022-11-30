#pragma once

#include "io/CommandQueue.h"
#include "win/D3DHeaders.h"

#include <unordered_map>

namespace Enso
{
    class UIGenericObject;
    
    class UIModuleInterface
    {
    public:
        UIModuleInterface(const std::string& id, std::shared_ptr<CommandQueue> outQueue) : m_componentId(id), m_outboundCmdQueue(outQueue) {}

        virtual void ConstructComponent() = 0;

        void SetInboundCommandQueue(std::shared_ptr<CommandQueue> inQueue) { m_inboundCmdQueue = inQueue; }

    protected:
        const std::string   m_componentId;

        std::unordered_map<std::string, std::shared_ptr<UIGenericObject>>   m_uiObjectMap;

        std::shared_ptr<CommandQueue>           m_inboundCmdQueue;
        std::shared_ptr<CommandQueue>           m_outboundCmdQueue;
    };
}