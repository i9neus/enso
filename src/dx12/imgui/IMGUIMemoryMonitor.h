#pragma once

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/JsonUtils.h"
#include "manager/RenderManager.h"

#include "shelves/IMGUIAbstractShelf.h"
#include <set>

class MemoryMonitor : public IMGUIElement
{
public:
    MemoryMonitor(const Json::Document& renderStateJson, Json::Document& commandQueue);

    void ConstructUI();

private:
    const Json::Document&       m_renderStateJson;
    Json::Document&             m_commandQueue;

    Json::Document              m_memoryStateJson;
    HighResolutionTimer         m_memoryStatsTimer;
};