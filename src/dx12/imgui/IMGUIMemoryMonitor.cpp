#include "IMGUIMemoryMonitor.h"

MemoryMonitor::MemoryMonitor(const Json::Document& renderStateJson, Json::Document& commandQueue) :
    m_renderStateJson(renderStateJson),
    m_commandQueue(commandQueue)
{
}

void MemoryMonitor::ConstructUI()
{
    ImGui::Begin("Memory Monitor");

    // Only poll the render object manager occasionally
    if (m_memoryStatsTimer.Get() > 0.5f)
    {
        // If we're waiting on a previous stats job, don't dispatch a new one
        if (!m_renderStateJson.GetChildObject("jobs/getMemoryStats", Json::kSilent))
        {
            m_commandQueue.AddChildObject("getMemoryStats");
            m_memoryStatsTimer.Reset();
        }
    }

    // Make a copy of the memory stats if any have been emitted
    const Json::Node statsJson = m_renderStateJson.GetChildObject("jobs/getMemoryStats", Json::kSilent);
    if (statsJson)
    {
        int statsState;
        statsJson.GetValue("state", statsState, Json::kRequiredAssert);

        // If the stats gathering task has finished, it'll be accompanied by data for each render object that emits it
        if (statsState == kRenderManagerJobCompleted)
        {
            const Json::Node assetJson = statsJson.GetChildObject("assets", Json::kSilent);
            m_memoryStateJson.DeepCopy(assetJson);
        }
    }

    IMGUIDataTable table("stats", 4);

    for (auto& it = m_memoryStateJson.begin(); it != m_memoryStateJson.end(); ++it)
    {
        int currBytes = 0, peakBytes = 0, deltaBytes = 0;
        const auto& asset = *it;
        asset.GetValue("currBytes", currBytes, Json::kSilent);
        asset.GetValue("peakBytes", peakBytes, Json::kSilent);
        asset.GetValue("deltaBytes", deltaBytes, Json::kSilent);
        table << it.Name() << currBytes << peakBytes << deltaBytes;
    }
    table.End();

    ImGui::End();
}