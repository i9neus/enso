#include "CudaAsset.cuh"
#include <map>

namespace Cuda
{
    GlobalAssetRegistry& GlobalAssetRegistry::Get()
    {
        static GlobalAssetRegistry singleton;
        return singleton;
    }

    void GlobalAssetRegistry::Register(std::shared_ptr<Host::Asset> object)
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        const std::string& assetId = object->GetAssetID();
        AssertMsgFmt(m_assetMap.find(assetId) == m_assetMap.end(), "Object '%s' is already in asset registry!", object->GetAssetID().c_str());
        
        m_assetMap.emplace(assetId, std::weak_ptr <Host::Asset>(object));
        Log::Debug("Registered asset '%s'.\n", assetId.c_str());
    }

    void GlobalAssetRegistry::Deregister(std::shared_ptr<Host::Asset> object)
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        const std::string& assetId = object->GetAssetID();
        if (m_assetMap.find(assetId) == m_assetMap.end())
        {
            Log::Error("Object '%s' does not exist in asset registry!\n", object->GetAssetID().c_str());
            return;
        }
        //AssertMsgFmt(m_assetMap.find(assetId) != m_assetMap.end(), "Object '%s' does not exist in asset registry!", object->GetAssetID().c_str());
        m_assetMap.erase(assetId);
    }

    void GlobalAssetRegistry::Report()
    {
        std::multimap<int, std::string> sortedAssets;
        for (auto& asset : m_assetMap)
        {
            if (asset.second.expired())
            {
                Log::Error("- WARNING: Registered asset '%s' expired without being removed from the registry. Was it explicitly destroyed?\n", asset.first.c_str());
                continue;
            }

            sortedAssets.emplace(asset.second.use_count(), asset.first);
        }

        for (auto& asset : sortedAssets)
        {
            if (asset.first == 1)
            {
                Log::Debug("- '%s' with %i references\n", asset.second, asset.first);
            }
            else
            {
                Log::Warning("- '%s' with %i references\n", asset.second, asset.first);
            }
        }
    }

    void GlobalAssetRegistry::VerifyEmpty()
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        if (m_assetMap.empty()) 
        { 
            Log::Write("SUCCESS! All managed assets were sucessfully cleaned up.\n");
            return; 
        }

        Log::Warning("WARNING: The following %i objects were not explicitly destroyed!\n", m_assetMap.size());
        
        Report(); 

        throw std::runtime_error("FAILED.");
    }

    bool GlobalAssetRegistry::Exists(const std::string& assetId) const
    {
        return m_assetMap.find(assetId) != m_assetMap.end();
    }
}

