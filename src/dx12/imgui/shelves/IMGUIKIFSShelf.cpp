#include "IMGUIKIFSShelf.h"
#include "generic/FilesystemUtils.h"

#include <random>

KIFSStateContainer::KIFSStateContainer()
{

}

void KIFSStateContainer::SetJsonPath(const std::string& filePath)
{
    m_jsonPath = filePath;
}

void KIFSStateContainer::ReadJson()
{
    Log::Debug("Trying to restore KIFS state library...\n");

    Json::Document rootDocument;
    try
    {
        rootDocument.Load(m_jsonPath);
    }
    catch (const std::runtime_error& err)
    {
        Log::Debug("Failed: %s.\n", err.what());
    }

    for (Json::Node::Iterator it = rootDocument.begin(); it != rootDocument.end(); ++it)
    {
        std::string newId = it.Name();
        Json::Node childNode = *it;

        Cuda::KIFSParams kifsParams(childNode, Json::kSilent);
        Insert(newId, kifsParams, false);
    }
}

void KIFSStateContainer::WriteJson()
{
    Json::Document rootDocument;
    for (auto& state : m_stateMap)
    {
        Json::Node childNode = rootDocument.AddChildObject(state.first);
        childNode.DeepCopy(*state.second);
    }

    rootDocument.WriteFile(m_jsonPath);
}

void KIFSStateContainer::Insert(const std::string& id, const Cuda::KIFSParams& kifsParams, bool overwriteIfExists)
{
    if (id.empty()) { Log::Error("Error: KIFS state ID must not be blank.\n");  return; }

    auto it = m_stateMap.find(id);
    std::shared_ptr<Json::Document> jsonPtr;
    if (it != m_stateMap.end()) 
    { 
        if (!overwriteIfExists) { Log::Error("Error: KIFS state with ID '%s' already exists.\n", id); return; }

        Assert(it->second);
        jsonPtr = it->second;
        jsonPtr->Clear();

        Log::Debug("Updated KIFS state '%s' in library.\n", id);
    }
    else
    {        
        auto& statePtr = m_stateMap[id];
        statePtr.reset(new Json::Document());
        jsonPtr = statePtr;

        Log::Debug("Added KIFS state '%s' to library.\n", id);
    }

    kifsParams.ToJson(*jsonPtr);
    WriteJson();

    //for (auto& e : m_stateMap) { Log::Debug("  - %s\n", e.first); }
}

void KIFSStateContainer::Erase(const std::string& id)
{
    auto it = m_stateMap.find(id);
    if (it == m_stateMap.end()) { Log::Error("Error: KIFS state with ID '%s' does not exist.\n", id); return; }

    m_stateMap.erase(it);
    WriteJson();

    Log::Debug("Removed KIFS state '%s' from library.\n", id);
}

void KIFSStateContainer::Restore(const std::string& id, Cuda::KIFSParams& kifsParams)
{
    auto it = m_stateMap.find(id);
    if (it == m_stateMap.end()) { Log::Error("Error: KIFS state with ID '%s' does not exist.\n", id); return; }

    Assert(it->second);
    kifsParams.FromJson(*(it->second), Json::kSilent);

    Log::Debug("Restored KIFS state '%s' from library.\n", id);
}

void KIFSStateContainer::ToJson(Json::Document& document)
{
}

KIFSShelf::KIFSShelf(const Json::Node& json) : IMGUIShelf(json)
{
    m_stateListCurrentIdx = -1;
    m_stateIDData.resize(2048);
    std::memset(m_stateIDData.data(), '\0', sizeof(char) * m_stateIDData.size());

    m_stateJsonPath = json.GetRootDocument().GetOriginFilePath();
    std::string jsonStem = GetFileStem(m_stateJsonPath);
    ReplaceFilename(m_stateJsonPath, tfm::format("%s.states.json", jsonStem));

    m_stateContainer.SetJsonPath(m_stateJsonPath);
    m_stateContainer.ReadJson();
}

void KIFSShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    auto& p = m_params[0];
    ConstructTransform(p.transform, true);

    const float TEXT_BASE_WIDTH = ImGui::CalcTextSize("A").x;

    auto ConstructRow = [](const std::string& label, Cuda::vec3& value, int row) -> void
    {
        ImGui::TableNextRow();
        ImGui::PushID(row);
        for (int column = 0; column < 2; column++)
        {
            ImGui::TableSetColumnIndex(column);
            if (column == 0)
            {
                ImGui::Text(label.c_str());
            }
            else
            {
                ImGui::PushItemWidth(140);
                ImGui::DragFloat("+/-", &value[0], 0.001f, 0.0f, 1.0f, "%.6f"); SL;
                ImGui::PopItemWidth();
                ImGui::PushItemWidth(80);
                ImGui::DragFloat("~", &value[1], math::max(0.00001f, value[1] * 0.01f), 0.0f, 1.0f, "%.6f"); SL;
                ImGui::SliderFloat("", &value[2], 0.0f, 1.0f);
                ImGui::PopItemWidth();
            }
        }
        ImGui::PopID();
    };

    if (ImGui::BeginTable("", 2))
    {
        // We could also set ImGuiTableFlags_SizingFixedFit on the table and all columns will default to ImGuiTableColumnFlags_WidthFixed.
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 500.0f);

        ConstructRow("Rotation A", p.rotateA, 0);
        ConstructRow("Rotation B", p.rotateB, 1);
        ConstructRow("Scale A", p.scaleA, 2);
        ConstructRow("Scale B", p.scaleB, 3);
        ConstructRow("Crust thickness", p.crustThickness, 4);
        ConstructRow("Vertex scale", p.vertScale, 5);

        ImGui::EndTable();
    }

    ImGui::SliderInt("Iterations ", &p.numIterations, 0, kSDFMaxIterations);
    ConstructComboBox("Fold type", std::vector<std::string>({ "Tetrahedron", "Cube" }), p.foldType);
    ConstructComboBox("Primitive type", std::vector<std::string>({ "Tetrahedron", "Cube" }), p.primitiveType);    

    // Jitter the current state to generate a new scene
    if (ImGui::Button("Randomise"))
    {
        p.Randomise();
    } 
    SL;
    // Reset all the jittered values to their midpoints
    if (ImGui::Button("Reset jitter"))
    {
        p.Randomise(0.5f, 0.5f);
    }
 
    if (ImGui::TreeNode("State manager"))
    {
        /*auto& stateMap = m_stateContainer.GetStateMap();
        if (ImGui::BeginListBox("States"))
        {
            KIFSStateContainer::StateMap::const_iterator it = stateMap.begin();
            for (int n = 0; n < stateMap.size(); n++, ++it)
            {
                const bool isSelected = (m_stateListCurrentIdx == n);
                if (ImGui::Selectable(it->first.c_str(), isSelected))
                {
                    m_stateListCurrentId = it->first;
                    m_stateListCurrentIdx = n;
                }
                if (isSelected) { ImGui::SetItemDefaultFocus(); }
            }
            ImGui::EndListBox();
        }
        ImGui::InputText("State ID", m_stateIDData.data(), m_stateIDData.size());

        // Save the current state to the container
        if (ImGui::Button("New"))
        {
            m_stateContainer.Insert(std::string(m_stateIDData.data()), m_params[0], false);
            std::memset(m_stateIDData.data(), '\0', sizeof(char) * m_stateIDData.size());
        }
        SL;
        // Overwrite the currently selected state
        if (ImGui::Button("Overwrite"))
        {
            if (m_stateListCurrentIdx < 0) { Log::Warning("Select a state from the list to overwrite it.\n"); }
            else
            {
                m_stateContainer.Insert(m_stateListCurrentId, m_params[0], true);                
            }
        }
        SL;
        // Load a saved state to the UI
        if (ImGui::Button("Load") && m_stateListCurrentIdx >= 0 && !stateMap.empty())
        {
            m_stateContainer.Restore(m_stateListCurrentId, m_params[0]);
        }
        SL;
        // Erase a saved state from the container
        if (ImGui::Button("Erase") && m_stateListCurrentIdx >= 0 && !stateMap.empty())
        {
            m_stateContainer.Erase(m_stateListCurrentId);
        }*/
        ImGui::TreePop();
    }

    auto ConstructMaskCheckboxes = [](const std::string& label, uint& value, const int row) -> void
    {
        ImGui::PushID(row);
        for (int i = 0; i < 6; i++)
        {
            bool faceMaskBool = value & (1 << i);
            ImGui::Checkbox(tfm::format("%i", i).c_str(), &faceMaskBool); SL;
            value = (value & ~(1 << i)) | (int(faceMaskBool) << i);
        }
        ImGui::PopID();
        ImGui::Text(label.c_str());
    };

    ConstructMaskCheckboxes("Face mask", p.faceMask.x, 0);
    ConstructMaskCheckboxes("Perturb", p.faceMask.y, 1);

    ImGui::Checkbox("SDF Clip Camera Rays", &p.sdf.clipCameraRays);
    ConstructComboBox("SDF Clip Shape", std::vector<std::string>({ "Cube", "Sphere", "Torus" }), p.sdf.clipShape);
    ImGui::DragInt("SDF Max Specular Interations", &p.sdf.maxSpecularIterations, 1, 1, 500);
    ImGui::DragInt("SDF Max Diffuse Iterations", &p.sdf.maxDiffuseIterations, 1, 1, 500);
    ImGui::DragFloat("SDF Cutoff Threshold", &p.sdf.cutoffThreshold, math::max(0.00001f, p.sdf.cutoffThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
    ImGui::DragFloat("SDF Escape Threshold", &p.sdf.escapeThreshold, math::max(0.01f, p.sdf.escapeThreshold * 0.01f), 0.0f, 5.0f);
    ImGui::DragFloat("SDF Ray Increment", &p.sdf.rayIncrement, math::max(0.01f, p.sdf.rayIncrement * 0.01f), 0.0f, 2.0f);
    ImGui::DragFloat("SDF Ray Kickoff", &p.sdf.rayKickoff, math::max(0.01f, p.sdf.rayKickoff * 0.01f), 0.0f, 1.0f);
    ImGui::DragFloat("SDF Fail Threshold", &p.sdf.failThreshold, math::max(0.00001f, p.sdf.failThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
}

void KIFSShelf::Reset()
{
}

void KIFSShelf::JitterKIFSParameters()
{
    
}
