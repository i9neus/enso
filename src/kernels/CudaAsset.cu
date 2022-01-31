#include "CudaAsset.cuh"
#include <map>

namespace Cuda
{
    GlobalResourceRegistry& GlobalResourceRegistry::Get()
    {
        static GlobalResourceRegistry singleton;
        return singleton;
    }

    void GlobalResourceRegistry::RegisterAsset(std::shared_ptr<Host::Asset> object)
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        const std::string& assetId = object->GetAssetID();
        AssertMsgFmt(m_assetMap.find(assetId) == m_assetMap.end(), "Object '%s' is already in asset registry!", object->GetAssetID().c_str());

        m_assetMap.emplace(assetId, std::weak_ptr <Host::Asset>(object));
        Log::Debug("Registered asset '%s'.\n", assetId.c_str());
    }

    void GlobalResourceRegistry::DeregisterAsset(std::shared_ptr<Host::Asset> object)
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        const std::string& assetId = object->GetAssetID();
        AssertMsgFmt(m_assetMap.find(assetId) == m_assetMap.end(), "Object '%s' does not exist in asset registry!", assetId.c_str());

        auto memoryIt = m_deviceMemoryMap.find(assetId);
        AssertMsgFmt(memoryIt == m_deviceMemoryMap.end(), "Object '%s' has %ll bytes of outstanding device memory that have not been deallocated.", assetId.c_str(), memoryIt->second);

        //AssertMsgFmt(m_assetMap.find(assetId) != m_assetMap.end(), "Object '%s' does not exist in asset registry!", object->GetAssetID().c_str());
        m_assetMap.erase(assetId);
    }

    void GlobalResourceRegistry::RegisterDeviceMemory(const std::string& assetId, const int64_t bytes)
    {
        Assert(bytes >= 0);
        if (bytes == 0) 
        { 
            Log::System("WARNING: Asset '%s' registered an allocation for zero bytes.", assetId);
            return; 
        }
        
        std::lock_guard<std::mutex> mutexLock(m_mutex);        

        auto& entry = m_deviceMemoryMap[assetId];
        entry += bytes;

        Log::System("Device alloc: %s allocated %ll bytes (%ll total).", assetId, bytes, int64_t(entry));
    }

    void GlobalResourceRegistry::DeregisterDeviceMemory(const std::string& assetId, const int64_t bytes)
    {
        Assert(bytes >= 0);
        if (bytes == 0) { return; }
        
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        auto it = m_deviceMemoryMap.find(assetId);
        AssertMsgFmt(it != m_deviceMemoryMap.end() && int64_t(it->second) - int64_t(bytes) >= 0ll, 
            "Asset '%s' is trying to deallocate more memory than it originally allocated.", assetId.c_str());

        // Decrement the allocated bytes and clean up the entry if necessary. 
        it->second -= bytes;
        if (it->second == 0ll) { m_deviceMemoryMap.erase(it); }
    }

    void GlobalResourceRegistry::Report()
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

    void GlobalResourceRegistry::VerifyEmpty()
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

    bool GlobalResourceRegistry::Exists(const std::string& assetId) const
    {
        return m_assetMap.find(assetId) != m_assetMap.end();
    }
}

