#pragma once

#include "Builder.cuh"
#include "core/containers/Object.cuh"

namespace Enso
{
    namespace Host
    {
        class BIH2DAsset : public Host::Asset
        {
        public:
            using NodeType = BIH2D::FullNode;

        public:
            __host__ BIH2DAsset(const Asset::InitCtx& initCtx, const uint& minBuildablePrims);
            __host__ virtual ~BIH2DAsset() noexcept;

            __host__ inline Host::Vector<uint>&     GetPrimitiveIndices() { Assert(m_hostIndices); return *m_hostIndices; }
            __host__ void                           Build(std::function<BBox2f(uint)>& functor, const bool printStats = false);
            __host__ const BIH2D::BIHData<NodeType>*     GetDeviceData() const { return m_deviceBIH.GetDeviceData(); }
            __host__ const BIH2D::BIHData<NodeType>&     GetHostData() const { return m_hostBIH; }

            //__host__ const BIH2DStats&              GetTreeStats() const { return m_stats; }
            //__host__ const Host::Vector<BIH2DFullNode>& GetHostNodes() const { return *m_hostNodes; }
            __host__ bool                           IsConstructed() const { return m_hostBIH.isConstructed; }
            __host__ BBox2f                         GetBoundingBox() const { return m_hostBIH.bBox; }

            std::function<void(const char*)>        m_debugFunctor = nullptr;

        private:
            __host__ void                           CheckTreeNodes() const;

        private:
            AssetHandle<Host::Vector<NodeType>>     m_hostNodes;
            AssetHandle<Host::Vector<uint>>         m_hostIndices;
            BIH2D::BIHData<NodeType>                m_hostBIH;
            Cuda::Object<BIH2D::BIHData<NodeType>>  m_deviceBIH;
            const uint                              m_minBuildablePrims;
            BIH2D::Stats                            m_stats;
        };
    }   
}