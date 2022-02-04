#include "IMGUIMemoryMonitor.h"

MemoryMonitor::MemoryMonitor(const Json::Document& renderStateJson, Json::Document& commandQueue) :
    m_renderStateJson(renderStateJson),
    m_commandQueue(commandQueue)
{
    m_host.peakMB = 0.0f;
}

void MemoryMonitor::AccumulateAllocMB(AssetNode& node)
{
    node.recursiveAllocMB = node.thisAllocMB;
    for (auto& child : node.childNodes)
    {
        AccumulateAllocMB(*child);
        node.recursiveAllocMB += child->recursiveAllocMB;
    }  

    // Sort each node's children according to its recursive allocated memory
    std::sort(node.childNodes.begin(), node.childNodes.end(), [](const AssetNode* lhs, const AssetNode* rhs)
        {
            return lhs->recursiveAllocMB > rhs->recursiveAllocMB;
        });
}

void MemoryMonitor::ParseMemoryStateJson()
{
    // Only poll the render object manager occasionally
    if (m_memoryStatsTimer.Get() > 0.5f)
    {
        // If we're waiting on a previous stats job, don't dispatch a new one
        if (m_renderStateJson.GetChildObject("jobs/getMemoryStats", Json::kSilent)) { return; }
        
        m_commandQueue.AddChildObject("getMemoryStats");
        m_memoryStatsTimer.Reset();
    }

    // Make a copy of the memory stats if any have been emitted
    const Json::Node statsJson = m_renderStateJson.GetChildObject("jobs/getMemoryStats", Json::kSilent);
    if (!statsJson) { return; }
    
    int statsState;
    statsJson.GetValue("state", statsState, Json::kRequiredAssert);

    // If the stats gathering task has finished, it'll be accompanied by data for each render object that emits it
    if (statsState != kRenderManagerJobCompleted) { return; }
    
    m_memoryStateJson.DeepCopy(statsJson);

    auto AppendLog = [](std::vector<float>& buffer, float& peakValue, const float newValue)
    {
        peakValue = max(peakValue, newValue);
        constexpr int kMaxLogSize = 1024;
        if (buffer.size() < kMaxLogSize)
        {
            buffer.push_back(newValue);
        }
        else
        {
            // TODO: Shunting the data like this sucks, but IMGUI doesn't support ring buffers. 
            // Use a better plotting library like ImPlot (https://github.com/epezent/implot)
            for (int i = 0; i < kMaxLogSize - 1; ++i) { buffer[i] = buffer[i + 1]; }
            buffer.back() = newValue;
        }
    };

    const Json::Node deviceJson = m_memoryStateJson.GetChildObject("device", Json::kSilent);
    m_device.hasData = false;
    if (deviceJson)
    {
        deviceJson.GetValue("name", m_device.name, Json::kRequiredAssert);
        deviceJson.GetValue("freeMB", m_device.freeMB, Json::kRequiredAssert);
        deviceJson.GetValue("totalMB", m_device.totalMB, Json::kRequiredAssert);
        m_device.allocMB = m_device.totalMB - m_device.freeMB;
        AppendLog(m_device.allocMBLog, m_device.peakMB, m_device.allocMB);

        m_device.memoryUsedPct = (m_device.totalMB - m_device.freeMB) / m_device.totalMB;
        m_device.hasData = true;
    }   

    const Json::Node hostJson = m_memoryStateJson.GetChildObject("host", Json::kSilent);
    m_host.hasData = false;
    if (hostJson)
    {
        hostJson.GetValue("totalMB", m_host.allocMB, Json::kRequiredAssert);
        AppendLog(m_host.allocMBLog, m_host.peakMB, m_host.allocMB);
        m_host.hasData = true;
    }

    const Json::Node assetJson = m_memoryStateJson.GetChildObject("assets", Json::kSilent);
    m_assets.hasData = false;
    if (assetJson)
    {
        m_assets.nodes.clear();
        m_assets.nodes.resize(assetJson.NumMembers());
        m_assets.nodeMap.clear();
        m_assets.rootNode.childNodes.clear();

        int nodeIdx = 0;
        for (auto& it = assetJson.begin(); it != assetJson.end(); ++it)
        {             
            // Create a new node and fill it with information
            const auto& asset = *it;            
            AssetNode& newNode = m_assets.nodes[nodeIdx++];
            newNode.id = it.Name();
            asset.GetValue("currMB", newNode.thisAllocMB, Json::kRequiredAssert);
            asset.GetValue("peakMB", newNode.peakMB, Json::kRequiredAssert);
            asset.GetValue("deltaMB", newNode.deltaMB, Json::kRequiredAssert);

            // Attach the node to its parent (if it has one)
            std::string parentId;
            asset.GetValue("parent", parentId, Json::kRequiredAssert);
            std::vector<AssetNode*>* parentNode = &m_assets.rootNode.childNodes;
            if (!parentId.empty())
            {
                auto& parentIt = m_assets.nodeMap.find(parentId);
                if (parentIt != m_assets.nodeMap.end())
                {
                    parentNode = &parentIt->second->childNodes;
                }
            }

            parentNode->push_back(&newNode);
            m_assets.nodeMap[newNode.id] = &newNode;
        }

        // Walk the tree and calculated the accumulated memory
        AccumulateAllocMB(m_assets.rootNode);        

        m_assets.hasData = true;
    }
}

void MemoryMonitor::ConstructAssetNode(const AssetNode& node, const bool isRoot)
{
    const auto ConstructAttrs = [&]()
    {       
        ImGui::TableNextColumn();
        ImGui::Text(FormatDataSize(node.recursiveAllocMB, 2).c_str());

        if (!isRoot)
        {
            ImGui::TableNextColumn();
            Text(FormatDataSize(node.thisAllocMB, 2).c_str());
            ImGui::TableNextColumn();
            Text(FormatDataSize(node.peakMB, 2).c_str());
            ImGui::TableNextColumn();
            Text(FormatDataSize(node.deltaMB, 2).c_str(), ImColor::HSV((node.deltaMB > 0.0f) ? 0.3f : 0.0f, 1.0f, 0.5f, 1.0f));
        }
    };
    
    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    if (!node.childNodes.empty())
    {
        const bool isOpen = ImGui::TreeNodeEx(isRoot ? "" : node.id.c_str(), ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_DefaultOpen);
        ConstructAttrs(); 
        if (isOpen)
        {
            for (const auto& childNode : node.childNodes)
            {
                ConstructAssetNode(*childNode, false);
            }           
            ImGui::TreePop();
        }
    }
    else
    {
        ImGui::TreeNodeEx(node.id.c_str(), ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_DefaultOpen);
        ConstructAttrs();
    }    
}

void MemoryMonitor::ConstructAssetTable()
{    
    static ImGuiTableFlags tableFlags = ImGuiTableFlags_BordersV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_Resizable | 
                                        ImGuiTableFlags_RowBg | ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_ScrollY;
    const auto kBaseSize = ImGui::CalcTextSize("A");

    if (!ImGui::BeginTable("assetTable", 5, tableFlags, ImVec2(0.0f, kBaseSize.y * 35.0f), 0.0f)) { return; }

    // The first column will use the default _WidthStretch when ScrollX is Off and _WidthFixed when ScrollX is On   
    ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_NoHide);
    ImGui::TableSetupColumn("GPU Total", ImGuiTableColumnFlags_WidthFixed, kBaseSize.x * 12.0f);
    ImGui::TableSetupColumn("GPU Object", ImGuiTableColumnFlags_WidthFixed, kBaseSize.x * 12.0f);
    ImGui::TableSetupColumn("GPU Object peak", ImGuiTableColumnFlags_WidthFixed, kBaseSize.x * 12.0f);
    ImGui::TableSetupColumn("GPU Object delta", ImGuiTableColumnFlags_WidthFixed, kBaseSize.x * 12.0f);
    ImGui::TableHeadersRow();

    ConstructAssetNode(m_assets.rootNode, true);

    ImGui::EndTable();
}

void MemoryMonitor::ConstructUI()
{
    ParseMemoryStateJson();
    
    ImGui::Begin("Memory Monitor");

    if(ImGui::TreeNodeEx("Device", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGuiIndent indent;
        if (m_device.hasData)
        {
            std::string memoryFmt = tfm::format("%.2fMB", m_device.allocMB);
            ImGui::PlotLines("Device memory", m_device.allocMBLog.data(), m_device.allocMBLog.size(), 0, memoryFmt.c_str(), 0, m_device.peakMB * 1.1f, ImVec2(0.0f, 50.0f));
            ImGui::ProgressBar(m_device.memoryUsedPct, ImVec2(0.0f, 0.0f)); SL; ImGui::Text("Capacity used");

            if (ImGui::Button("Reset"))
            {
                m_device.allocMBLog.clear();
                m_device.peakMB = 0.0f;
            }

            /*if (ImGui::TreeNodeEx("Raw JSON", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::TextWrapped(deviceJson.Stringify(true).c_str());
            }*/

            IMGUIDataTable table("device", 2); 
            table << "Device name" << m_device.name;
            table << "Allocated" << FormatDataSize(m_device.allocMB, 2);
            table << "Peak" << FormatDataSize(m_device.peakMB, 2);
            table << "Total available" << FormatDataSize(m_device.totalMB, 2);
            table.End();
        }
        else
        {
            Text("No data", ImVec4(ImColor::HSV(1.0f, 1.0f, 0.5f)));
        }

        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Host", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGuiIndent indent;
        if (m_host.hasData)
        {
            std::string memoryFmt = tfm::format("%.2fMB", m_host.allocMBLog.back());
            ImGui::PlotLines("Host memory", m_host.allocMBLog.data(), m_host.allocMBLog.size(), 0, memoryFmt.c_str(), 0, m_host.peakMB * 1.1f, ImVec2(0.0f, 50.0f));

            if (ImGui::Button("Reset"))
            {
                m_host.allocMBLog.clear();
                m_host.peakMB = 0.0f;
            }

            IMGUIDataTable table("host", 2);
            table << "Allocated" << FormatDataSize(m_host.allocMB, 2);
            table << "Peak" << FormatDataSize(m_host.peakMB, 2);
            table.End();
        }
        else
        {
            Text("No data", ImVec4(ImColor::HSV(1.0f, 1.0f, 0.5f)));
        }

        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Assets", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGuiIndent indent;
        if (m_assets.hasData)
        {
            ConstructAssetTable();
        }
        else
        {
            Text("No data", ImVec4(ImColor::HSV(1.0f, 1.0f, 0.5f)));
        }

        ImGui::TreePop();
    }

    ImGui::End();
}