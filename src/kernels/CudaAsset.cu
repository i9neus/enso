#include "CudaAsset.cuh"

namespace Cuda
{
    GlobalAssetRegistry& GlobalAssetRegistry::Get()
    {
        static GlobalAssetRegistry singleton;
        return singleton;
    }

    void GlobalAssetRegistry::Register(std::shared_ptr<AssetBase> object)
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        AssertMsgFmt(m_assetMap.find(object.get()) == m_assetMap.end(), "Object '%s' is already in asset registry!", object->GetAssetName().c_str())
            m_assetMap.emplace(object.get(), std::weak_ptr <AssetBase>(object));
    }

    void GlobalAssetRegistry::Deregister(std::shared_ptr<AssetBase> object)
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        AssertMsgFmt(m_assetMap.find(object.get()) != m_assetMap.end(), "Object '%s' does not exist in asset registry!", object->GetAssetName().c_str());
        m_assetMap.erase(object.get());
    }

    void GlobalAssetRegistry::VerifyEmpty()
    {
        std::lock_guard<std::mutex> mutexLock(m_mutex);

        if (m_assetMap.empty()) { return; }

        std::printf("WARNING: The following %i objects were not explicitly destroyed!\n", m_assetMap.size());
        for (auto& asset : m_assetMap)
        {
            Assert(!asset.second.expired(), "A registered asset has expired without being removed from the registry. This should never happen!");

            AssetBase* object = asset.first;
            Assert(object);

            std::printf("  - '%s' with %i ref counts\n", object->GetAssetName().c_str(), asset.second.use_count());
        }

        throw std::runtime_error("Memory leak.");
    }
}

