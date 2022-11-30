#pragma once

#include <memory>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>

namespace Enso
{
    namespace Json { class Node; }
    namespace Host { class Asset; }
    
    class GlobalResourceRegistry
    {
    public:
        struct MemoryStats
        {
            int64_t     currentBytes = 0;
            int64_t     peakBytes = 0;
            int64_t     deltaBytes = 0;
        };

    public:
        static GlobalResourceRegistry& Get();

        void RegisterAsset(std::weak_ptr<Host::Asset> object, const std::string& assetId);
        void DeregisterAsset(std::weak_ptr<Host::Asset> object, const std::string& assetId);
        void RegisterDeviceMemory(const std::string& assetId, const int64_t bytes);
        void DeregisterDeviceMemory(const std::string& assetId, const int64_t bytes);
        void VerifyEmpty();
        void Report();
        bool Exists(const std::string& id) const;

        const std::unordered_map<std::string, std::weak_ptr<Host::Asset>>& GetAssetMap() const { return m_assetMap; }
        const std::unordered_map<std::string, MemoryStats>& GetDeviceMemoryMap() const { return m_deviceMemoryMap; }
        size_t NumAssets() const { return m_assetMap.size(); }

    private:
        GlobalResourceRegistry() = default;

        std::unordered_map<std::string, std::weak_ptr<Host::Asset>>     m_assetMap;
        std::unordered_map<std::string, MemoryStats>                    m_deviceMemoryMap;
        std::mutex                                                      m_mutex;
    };

    static inline GlobalResourceRegistry& AR() { return GlobalResourceRegistry::Get(); }
}