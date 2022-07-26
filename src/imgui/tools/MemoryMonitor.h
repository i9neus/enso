#pragma once

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/JsonUtils.h"
#include "manager/RenderManager.h"

#include "shelves/IMGUIAbstractShelf.h"
#include <set>

class MemoryMonitor : public IMGUIWidget
{
public:
    MemoryMonitor(const Json::Document& renderStateJson, Json::Document& commandQueue);

    void ConstructUI();

private:
    struct AssetNode
    {
        std::string             id;
        float                   thisAllocMB;
        float                   recursiveAllocMB;
        float                   peakMB;
        float                   deltaMB;

        std::vector<AssetNode*> childNodes;
    };

    void AccumulateAllocMB(AssetNode& node);
    void ParseMemoryStateJson();
    void ConstructAssetTable();
    void ConstructAssetNode(const AssetNode& node, const bool isRoot);

    const Json::Document&       m_renderStateJson;
    Json::Document&             m_commandQueue;

    Json::Document              m_memoryStateJson;
    HighResolutionTimer         m_memoryStatsTimer;
     
    struct
    {
        bool                    hasData = false;
        std::vector<float>      allocMBLog;
        float                   allocMB;
        float                   peakMB;
    }
    m_host;

    struct
    {
        bool                    hasData = false;
        std::string             name;
        std::vector<float>      allocMBLog;
        float                   freeMB;
        float                   allocMB;
        float                   totalMB;
        float                   peakMB;
        float                   memoryUsedPct;
    }
    m_device;    

    struct
    {
        bool                                        hasData = false;
        std::vector<AssetNode>                      nodes;
        std::unordered_map<std::string, AssetNode*> nodeMap;
        AssetNode                                   rootNode;
    }
    m_assets;
};