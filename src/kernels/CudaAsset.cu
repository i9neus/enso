#include "CudaAsset.cuh"

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

        const std::string& assetName = object->GetAssetName();
        AssertMsgFmt(m_assetMap.find(assetName) == m_assetMap.end(), "Object '%s' is already in asset registry!", object->GetAssetName().c_str());
        
        m_assetMap.emplace(assetName, std::weak_ptr <Host::Asset>(object));
        std::printf("Registered asset '%s'.\n", assetName.c_str());
    }

    void GlobalAssetRegistry::Deregister(std::shared_ptr<Host::Asset> object)
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        const std::string& assetName = object->GetAssetName();
        AssertMsgFmt(m_assetMap.find(assetName) != m_assetMap.end(), "Object '%s' does not exist in asset registry!", object->GetAssetName().c_str());
        m_assetMap.erase(assetName);
    }

    void GlobalAssetRegistry::Report()
    {
        for (auto& asset : m_assetMap)
        {
            if (asset.second.expired())
            {
                std::printf("  - WARNING: Registered asset '%s' expired without being removed from the registry. Was it explicitly destroyed?\n", asset.first.c_str());
                continue;
            }

            std::printf("  - '%s' with %i ref counts\n", asset.first.c_str(), asset.second.use_count());
        }
    }

    void GlobalAssetRegistry::VerifyEmpty()
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        if (m_assetMap.empty()) 
        { 
            std::printf("SUCCESS! All managed assets were sucessfully cleaned up.\n");
            return; 
        }

        std::printf("WARNING: The following %i objects were not explicitly destroyed!\n", m_assetMap.size());
        
        Report(); 

        throw std::runtime_error("FAILED.");
    }
}

