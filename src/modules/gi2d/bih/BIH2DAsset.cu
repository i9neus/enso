#include "BIH2DAsset.cuh"

namespace Enso
{
    __device__ void Device::BIH2DAsset::Synchronise(const BIH2DParams<BIH2DFullNode>& params)
    {
        m_nodes = params.nodes->Data();
        m_numNodes = params.nodes->Size();
        m_numPrims = params.numPrims;
        m_treeBBox = params.bBox;
        m_isConstructed = params.isConstructed;
        m_testAsList = params.testAsList;
    }

    __host__ Host::BIH2DAsset::BIH2DAsset(const std::string& id, const uint& minBuildablePrims) :
        Asset(id),
        m_allocator(*this),
        cu_deviceInstance(nullptr),
        m_minBuildablePrims(minBuildablePrims)
    {
        cu_deviceInstance = m_allocator.InstantiateOnDevice<Device::BIH2DAsset>();
        cu_deviceInterface = m_allocator.StaticCastOnDevice<BIH2D<BIH2DFullNode>>(cu_deviceInstance);

        m_hostNodes = m_allocator.CreateChildAsset<Host::Vector<NodeType>>(tfm::format("%s_nodes", id), kVectorHostAlloc);
    }

    __host__ Host::BIH2DAsset::~BIH2DAsset()
    {
        OnDestroyAsset();
    }

    void Host::BIH2DAsset::OnDestroyAsset()
    {
        m_hostNodes.DestroyAsset();

        m_allocator.DestroyOnDevice(cu_deviceInstance);
    }

    __host__ void Host::BIH2DAsset::Build(std::function<BBox2f(uint)>& functor)
    {
        Assert(m_hostNodes);
        //AssertMsg(!m_primitiveIdxs.empty(), "BIH builder does not contain any primitives.");

        // Resize and reset pointers
        BIH2DBuilder<NodeType> builder(*this, *m_hostNodes, m_primitiveIdxs, m_minBuildablePrims, functor);
        builder.m_debugFunctor = m_debugFunctor;
        builder.Build();

        //CheckTreeNodes();

        // Synchronise the node data to the device
        m_hostNodes->Synchronise(kVectorSyncUpload);
        m_params.isConstructed = m_isConstructed;
        m_params.testAsList = m_testAsList;
        m_params.bBox = m_treeBBox;
        m_params.nodes = m_hostNodes->GetDeviceInstance();
        m_params.numPrims = uint(m_primitiveIdxs.size());

        SynchroniseObjects<Device::BIH2DAsset>(cu_deviceInstance, m_params);
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