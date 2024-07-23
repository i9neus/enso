#include "BIH2DAsset.cuh"

namespace Enso
{
    __device__ void Device::BIH2DAsset::Synchronise(const BIH2DData<BIH2DFullNode>& data)
    {
        m_nodes = data.nodes->Data();
        m_indices = data.indices->Data();

        m_numNodes = data.nodes->Size();
        m_numPrims = data.numPrims;
        m_treeBBox = data.bBox;
        m_isConstructed = data.isConstructed;
        m_testAsList = data.testAsList;
    }

    __host__ Host::BIH2DAsset::BIH2DAsset(const Asset::InitCtx& initCtx, const uint& minBuildablePrims) :
        Asset(initCtx),
        cu_deviceInstance(nullptr),
        m_minBuildablePrims(minBuildablePrims)
    {
        cu_deviceInstance = AssetAllocator::InstantiateOnDevice<Device::BIH2DAsset>(*this);

        m_hostNodes = AssetAllocator::CreateChildAsset<Host::Vector<NodeType>>(*this, "nodes", kVectorHostAlloc);
        m_hostIndices = AssetAllocator::CreateChildAsset<Host::Vector<uint>>(*this, "indices", kVectorHostAlloc);
    }

    __host__ Host::BIH2DAsset::~BIH2DAsset() noexcept
    {
        m_hostNodes.DestroyAsset();
        m_hostIndices.DestroyAsset();

        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::BIH2DAsset::Build(std::function<BBox2f(uint)>& functor)
    {
        Assert(m_hostNodes);
        //AssertMsg(!m_primitiveIdxs.empty(), "BIH builder does not contain any primitives.");

        // Resize and reset pointers
        BIH2DBuilder<NodeType> builder(*this, *m_hostNodes, *m_hostIndices, m_minBuildablePrims, functor);
        builder.m_debugFunctor = m_debugFunctor;
        builder.Build();

        //CheckTreeNodes();

        // Synchronise the node data to the device
        m_hostNodes->Synchronise(kVectorSyncUpload);
        m_hostIndices->Synchronise(kVectorSyncUpload);

        m_data.isConstructed = m_isConstructed;
        m_data.testAsList = m_testAsList;
        m_data.bBox = m_treeBBox;
        m_data.nodes = m_hostNodes->GetDeviceInstance();
        m_data.indices = m_hostIndices->GetDeviceInstance();
        m_data.numPrims = uint(m_hostIndices->Size());

        SynchroniseObjects<Device::BIH2DAsset>(cu_deviceInstance, m_data);
    }

    __host__ void Host::BIH2DAsset::CheckTreeNodes() const
    {
        for (int nodeIdx = 0; nodeIdx < m_hostNodes->Size(); ++nodeIdx)
        {
            const BIH2DFullNode& node = (*m_hostNodes)[nodeIdx];
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
                    if (plane < m_treeBBox.lower[axis] || plane > m_treeBBox.upper[axis])
                    {
                        Log::Error("Plane %i at node %i lies outside the tree bounding box: %f -> [%f, %f]", planeIdx, nodeIdx, plane, m_treeBBox.lower[axis], m_treeBBox.upper[axis]);
                    }
                }
            }
        }
    }
}