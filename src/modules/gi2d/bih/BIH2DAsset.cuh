#pragma once

#include "BIH2DBuilder.cuh"

namespace Enso
{
    namespace Device
    {
        class BIH2DAsset : public BIH2D<BIH2DFullNode>, public Device::Asset
        {
        public:
            __device__ BIH2DAsset() {}

            __device__ void Synchronise(const BIH2DParams<BIH2DFullNode>& params);
        };
    }

    namespace Host
    {
        class BIH2DAsset : public BIH2D<BIH2DFullNode>, public Host::Asset
        {
            using NodeType = BIH2DFullNode;
            using SubclassType = BIH2D<NodeType>;
        public:
            __host__ BIH2DAsset(const std::string& id, const uint& minBuildablePrims);
            __host__ virtual ~BIH2DAsset();

            __host__ virtual void                   OnDestroyAsset() override final;

            __host__ inline std::vector<uint>& GetPrimitiveIndices() { return m_primitiveIdxs; }
            __host__ void                           Build(std::function<BBox2f(uint)>& functor);
            __host__ Device::BIH2DAsset* GetDeviceInstance() const { return cu_deviceInstance; }
            __host__ BIH2D<BIH2DFullNode>* GetDeviceInterface() const { return cu_deviceInterface; }

            __host__ void                           Synchronise();
            __host__ const BIH2DStats& GetTreeStats() const { return m_stats; }
            __host__ const Host::Vector<BIH2DFullNode>& GetHostNodes() const { return *m_hostNodes; }

            std::function<void(const char*)> m_debugFunctor = nullptr;

        private:
            __host__ void                           CheckTreeNodes() const;

        private:
            AssetHandle<Host::Vector<BIH2DFullNode>> m_hostNodes;
            std::vector<uint>                       m_primitiveIdxs;
            BIH2DParams<BIH2DFullNode>              m_params;
            const uint                              m_minBuildablePrims;
            AssetAllocator                          m_allocator;

            Device::BIH2DAsset* cu_deviceInstance;
            BIH2D<BIH2DFullNode>* cu_deviceInterface;
        };
    }   
}