#include "BIHAsset.cuh"

namespace Enso
{
    __host__ Host::BIH2DAsset::BIH2DAsset(const Asset::InitCtx& initCtx, const uint& minBuildablePrims) :
        Asset(initCtx),
        m_minBuildablePrims(minBuildablePrims)
    {
        m_hostNodes = AssetAllocator::CreateChildAsset<Host::Vector<NodeType>>(*this, "nodes");
        m_hostIndices = AssetAllocator::CreateChildAsset<Host::Vector<uint>>(*this, "indices");
    }

    __host__ Host::BIH2DAsset::~BIH2DAsset() noexcept
    {
        m_hostNodes.DestroyAsset();
        m_hostIndices.DestroyAsset();
    }

    __host__ void Host::BIH2DAsset::Build(std::function<BBox2f(uint)>& functor, const bool printStats)
    {
        Assert(m_hostNodes);
        //AssertMsg(!m_primitiveIdxs.empty(), "BIH builder does not contain any primitives.");

        // Resize and reset pointers
        BIH2D::Builder<NodeType> builder(m_hostBIH, *m_hostNodes, *m_hostIndices, m_minBuildablePrims, functor);
        builder.m_debugFunctor = m_debugFunctor;
        builder.Build(printStats);

        //CheckTreeNodes();

        // Synchronise the node data to the device
        m_hostNodes->Upload();
        m_hostIndices->Upload();

        m_deviceBIH = m_hostBIH;
        m_deviceBIH->nodes = m_hostNodes->GetDeviceData();
        m_deviceBIH->indices = m_hostIndices->GetDeviceData();          
        m_deviceBIH.Upload();
    }

    __host__ void Host::BIH2DAsset::CheckTreeNodes() const
    {
        for (int nodeIdx = 0; nodeIdx < m_hostNodes->size(); ++nodeIdx)
        {
            const BIH2D::FullNode& node = (*m_hostNodes)[nodeIdx];
            if (node.IsLeaf())
            {
            }
            else
            {
                for (int planeIdx = 0; planeIdx < 2; ++planeIdx)
                {
                    const float plane = node.data.planes[planeIdx];
                    if (!std::isfinite(plane))
                    {
                        Log::Error("Plane %i at node %i is non-finite: %f", planeIdx, nodeIdx, plane);
                    }
                    if (!std::isnormal(plane))
                    {
                        Log::Error("Plane %i at node %i is denormalised: %f", planeIdx, nodeIdx, plane);
                    }
                    const uchar axis = node.GetAxis();
                    if (plane < m_hostBIH.bBox.lower[axis] || plane > m_hostBIH.bBox.upper[axis])
                    {
                        Log::Error("Plane %i at node %i lies outside the tree bounding box: %f -> [%f, %f]", planeIdx, nodeIdx, plane, m_hostBIH.bBox.lower[axis], m_hostBIH.bBox.upper[axis]);
                    }
                }
            }
        }
    }
}