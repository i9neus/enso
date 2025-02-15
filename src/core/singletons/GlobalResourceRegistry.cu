#include "GlobalResourceRegistry.cuh"
#include "core/math/Hash.cuh"
#include "thirdparty/tinyformat/tinyformat.h"
#include <map>
#include "io/Log.h"
#include "core/debug/Assert.h"

namespace Enso
{
    GlobalResourceRegistry& GlobalResourceRegistry::Get()
    {
        static GlobalResourceRegistry singleton;
        return singleton;
    }

    void GlobalResourceRegistry::RegisterAsset(std::weak_ptr<Host::Asset> object, const std::string& assetId)
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        AssertMsgFmt(m_assetMap.find(assetId) == m_assetMap.end(), "Object '%s' is already in asset registry!", assetId.c_str());

        m_assetMap.emplace(assetId, object);
        Log::Debug("Registered asset '%s'.\n", assetId.c_str());
    }

    void GlobalResourceRegistry::DeregisterAsset(const std::string& assetId) noexcept
    {
        // NOTE: By this point, the asset ought to have been deleted so the weak pointer in the asset map will have expired.
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        if (m_assetMap.find(assetId) == m_assetMap.end())
        {
            m_errorList.push_back(tinyformat::format("Deregister asset: object '%s' does not exist in the registry!", assetId));
            return;
        }

        auto memoryIt = m_deviceMemoryMap.find(assetId);
        if (memoryIt != m_deviceMemoryMap.end())
        {
            m_errorList.push_back(tinyformat::format("ERROR: Deregister asset: object '%s' has %i bytes of outstanding device memory that have not been deallocated.", assetId, memoryIt->second.currentBytes));
            return;
        }

        m_deviceMemoryMap.erase(assetId);
        m_assetMap.erase(assetId);
    }

    void GlobalResourceRegistry::RegisterDeviceMemory(const std::string& assetId, const int64_t newBytes)
    {
        Assert(newBytes >= 0);
        if (newBytes == 0)
        {
            Log::System("WARNING: Asset '%s' registered an allocation for zero bytes.", assetId);
            return;
        }

        std::lock_guard<std::mutex> mutexLock(m_mutex);

        auto& entry = m_deviceMemoryMap[assetId];
        entry.currentBytes += newBytes;
        entry.peakBytes = max(entry.peakBytes, entry.currentBytes);
        entry.deltaBytes = newBytes;

        Log::System("*** DEVICE ALLOC *** : %s -> %i bytes (%i in total)", assetId, newBytes, int64_t(entry.currentBytes));
    }

    void GlobalResourceRegistry::DeregisterDeviceMemory(const std::string& assetId, const int64_t delBytes) noexcept
    {
        Assert(delBytes >= 0);
        if (delBytes == 0) { return; }

        std::lock_guard<std::mutex> mutexLock(m_mutex);

        auto it = m_deviceMemoryMap.find(assetId);
        if (it == m_deviceMemoryMap.end())
        {
            m_errorList.push_back(tinyformat::format("Error: Asset '%s' is not in the registry", assetId));
            return;
        }
        auto& stats = it->second;

        Log::System("*** DEVICE FREE *** : %s -> %i (%i bytes remaining)", assetId, delBytes, int64_t(stats.currentBytes) - delBytes);
        
        if (int64_t(stats.currentBytes) - delBytes < 0)
        {
            m_errorList.push_back(tinyformat::format("Asset '%s' tried to deallocate more memory than it originally allocated.", assetId));
            return;
        }

        // Decrement the allocated bytes and clean up the entry if necessary. 
        stats.currentBytes -= delBytes;
        stats.deltaBytes = -delBytes;
        if (stats.currentBytes == 0) { m_deviceMemoryMap.erase(it); }
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
                Log::Warning("- '%s' with %i references\n", asset.second, asset.first);
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

        if (m_assetMap.empty() && m_errorList.empty())
        {
            Log::Write("SUCCESS! All managed assets were sucessfully cleaned up.\n");
            return;
        }

        if (!m_assetMap.empty())
        {
            Log::Error("WARNING: The following %i objects were not explicitly destroyed:\n", m_assetMap.size());
            Report();
        }

        if (!m_errorList.empty())
        {
            Log::Error("WARNING: The following objects reported errors:\n", m_assetMap.size());
            for (const auto& error : m_errorList)
            {
                Log::Error("  - %s", error);
            }
        }

        AssertMsg(m_assetMap.empty(), "Scene unloading resulted in errors.");
    }

    bool GlobalResourceRegistry::Exists(const std::string& assetId) const
    {
        return m_assetMap.find(assetId) != m_assetMap.end();
    }
}