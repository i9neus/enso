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

            __device__ void Synchronise(const BIH2DData<BIH2DFullNode>& params);
        };
    }

    namespace Host
    {
        class BIH2DAsset : public BIH2D<BIH2DFullNode>, public Host::Asset
        {
            using NodeType = BIH2DFullNode;
            using SubclassType = BIH2D<NodeType>;
        public:
            __host__ BIH2DAsset(const Asset::InitCtx& initCtx, const uint& minBuildablePrims);
            __host__ virtual ~BIH2DAsset() noexcept;

            __host__ inline Host::Vector<uint>&     GetPrimitiveIndices() { Assert(m_hostIndices); return *m_hostIndices; }
            __host__ void                           Build(std::function<BBox2f(uint)>& functor, const bool printStats = false);
            __host__ Device::BIH2DAsset*            GetDeviceInstance() const { return cu_deviceInstance; }

            __host__ const BIH2DStats&              GetTreeStats() const { return m_stats; }
            __host__ const Host::Vector<BIH2DFullNode>& GetHostNodes() const { return *m_hostNodes; }

            std::function<void(const char*)>        m_debugFunctor = nullptr;

        private:
            __host__ void                           CheckTreeNodes() const;

        private:
            AssetHandle<Host::Vector<BIH2DFullNode>> m_hostNodes;
            AssetHandle<Host::Vector<uint>>         m_hostIndices;
            BIH2DData<BIH2DFullNode>                m_data;
            const uint                              m_minBuildablePrims;

            Device::BIH2DAsset*                     cu_deviceInstance;
        };
    }   
}