#pragma once

#include "shelves/IMGUIAbstractShelf.h"

namespace Cuda
{
    namespace Host { class LightProbeCamera; }
    class RenderObjectContainer;
}

class RenderObjectStateManager : public IMGUIElement
{
public:
    using StateMap = std::map<const std::string, std::shared_ptr<Json::Document>>;

    RenderObjectStateManager(IMGUIAbstractShelfMap& imguiShelves);

    void Initialise(const Json::Node& node, const Cuda::AssetHandle<Cuda::RenderObjectContainer>& renderObjects);
    
    void ConstructUI();

    void ReadJson();
    void WriteJson();

    bool Insert(const std::string& id, const bool overwriteIfExists);
    bool Erase(const std::string& id);
    bool Restore(const std::string& id);
    void Clear();

    const StateMap& GetStateMap() const { return m_stateMap; }

private:
    void ConstructStateManagerUI();
    void ConstructBatchProcessorUI();

    StateMap                m_stateMap;
    IMGUIAbstractShelfMap&  m_imguiShelves;
    Cuda::AssetHandle<Cuda::Host::LightProbeCamera>   m_lightProbeCameraAsset;

    int                     m_numPermutations;
    bool                    m_isBaking;

    std::string             m_stateJsonPath;

    IMGUIListBox            m_sampleCountList;
    IMGUIListBox            m_stateList;
};