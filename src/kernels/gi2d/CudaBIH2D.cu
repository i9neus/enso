#include "CudaBIH2D.cuh"

namespace GI2D
{
    using namespace Cuda;
    
    __device__ void Device::BIH2DAsset::Synchronise(const BIH2DParams<BIH2DFullNode>& params)
    {
        assert(params.nodes);
        m_nodes = params.nodes->Data();
        m_numNodes = params.nodes->Size();
        m_numPrims = params.numPrims;
        m_treeBBox = params.bBox;
        m_isConstructed = params.isConstructed;
        m_testAsList = params.testAsList;
    }
    
    __host__ Host::BIH2DAsset::BIH2DAsset(const std::string& id) : 
        Asset(id),
        cu_deviceInstance(nullptr)
    {
        cu_deviceInstance = InstantiateOnDevice<Device::BIH2DAsset>(GetAssetID());

        m_hostNodes = CreateChildAsset<Cuda::Host::Vector<NodeType>>(tfm::format("%s_nodes", id), this, kVectorHostAlloc, m_hostStream);
    }

    __host__ Host::BIH2DAsset::~BIH2DAsset()
    {
        OnDestroyAsset();
    }

    void Host::BIH2DAsset::OnDestroyAsset()
    {
        m_hostNodes.DestroyAsset();

        DestroyOnDevice(GetAssetID(), cu_deviceInstance);
    }

    __host__ Host::BIH2DBuilder::BIH2DBuilder(Host::BIH2DAsset& bih, std::vector<uint>& primitiveIdxs, const uint minBuildablePrims, std::function<BBox2f(uint)>& functor) noexcept :
        m_bih(bih),
        m_hostNodes(*bih.m_hostNodes),
        m_primitiveIdxs(primitiveIdxs),
        m_getPrimitiveBBox(functor),
        m_stats(bih.m_stats),
        m_minBuildablePrims(minBuildablePrims)
    {
    }

    //template<typename NodeContainer>
    __host__ void Host::BIH2DBuilder::Build()
    {
        Timer timer;
        AssertMsg(m_getPrimitiveBBox, "BIH builder does not have a valid bounding box functor.");

        // Find the global bounding box
        BBox2f centroidBBox = BBox2f::MakeInvalid();
        m_bih.m_treeBBox = BBox2f::MakeInvalid();

        if (!m_primitiveIdxs.empty())
        {
            for (const auto idx : m_primitiveIdxs)
            {
                const BBox2f primBBox = m_getPrimitiveBBox(idx);
                AssertMsgFmt(primBBox.HasValidArea(),
                    "BIH primitive at index %i has returned an invalid bounding box: {%s, %s}", primBBox[0].format().c_str(), primBBox[1].format().c_str());
                m_bih.m_treeBBox = Union(m_bih.m_treeBBox, primBBox);
                centroidBBox = Union(centroidBBox, primBBox.Centroid());
            }

            AssertMsgFmt(m_bih.m_treeBBox.HasValidArea() && !m_bih.m_treeBBox.IsInfinite(),
                "BIH bounding box is invalid: %s", m_bih.m_treeBBox.Format().c_str());
        }

        // If the list contains below the minimum ammount of primitives, don't build and flag the tree as a list traversal
        if (m_primitiveIdxs.size() < m_minBuildablePrims)
        {
            m_bih.m_testAsList = true;
        }
        else
        {
            // Reserve space for the nodes
            m_hostNodes.Reserve(m_primitiveIdxs.size());
            m_hostNodes.Resize(1);

            // Construct the bounding interval hierarchy
            m_stats = BIH2DStats();
            m_stats.numInnerNodes = 0;

            BuildPartition(0, m_primitiveIdxs.size(), 0, 0, m_bih.m_treeBBox, centroidBBox);
        }

        // Update the host data structures
        m_bih.m_nodes = m_hostNodes.GetHostData();        
        m_bih.m_numNodes = m_hostNodes.Size();
        m_bih.m_numPrims = m_primitiveIdxs.size();
        m_bih.m_isConstructed = true;
        
        m_stats.buildTime = timer.Get() * 1e-3f;
        Log::Write("Constructed BIH in %.1fms", m_stats.buildTime);
        Log::Write("  - Max depth: %i", m_stats.maxDepth);
        Log::Write("  - Inner nodes: %i", m_stats.numInnerNodes);
        Log::Write("  - Leaf nodes: %i", m_stats.numLeafNodes);
    }

    //template<typename NodeContainer>
    __host__ void Host::BIH2DBuilder::BuildPartition(const int i0, const int i1, const uint thisIdx, const uchar depth, const BBox2f& parentBBox, const BBox2f& centroidBBox)
    {    
        // Sanity checks
        Assert(depth < 16);
        
        m_stats.maxDepth = max(depth, m_stats.maxDepth);

        // If this node only contains one primitive, it's a leaf
        if (i1 - i0 <= 1)
        {
            if (i0 == i1)
            {
                m_stats.numLeafNodes++;
                m_hostNodes[thisIdx].MakeLeaf(BIH2DFullNode::kInvalidLeaf, BIH2DFullNode::kInvalidLeaf);
                //return;
            }
            else if (i1 - i0 == 1)
            {
                m_stats.numLeafNodes++;
                m_hostNodes[thisIdx].MakeLeaf(m_primitiveIdxs[i0], m_primitiveIdxs[i1 - 1]);
                //return;
            }

            /*if (m_debugFunctor)
            {
                m_debugFunctor("----------\n");
                m_debugFunctor(tfm::format("Leaf: %i (%i)\n", thisIdx, m_hostNodes[thisIdx].data).c_str());
                for (int idx = 0; idx < m_hostNodes.Size(); ++idx)
                {
                    if(m_hostNodes[idx].IsLeaf())
                        m_debugFunctor(tfm::format("  %i -> Leaf %i\n", idx, m_hostNodes[idx].data.primIdxs[0]).c_str());
                    else
                        m_debugFunctor(tfm::format("  %i -> Inner %i, %i\n", idx, m_hostNodes[idx].GetAxis(), m_hostNodes[idx].GetChildIndex()).c_str());
                }
                m_debugFunctor("\n");
            }*/
            return;
        }

        Assert(parentBBox.HasPositiveArea());

        const uint axis = centroidBBox.MaxAxis();
        const float split = centroidBBox.Centroid(axis);
        BBox2f leftBBox = parentBBox, rightBBox = parentBBox;
        BBox2f leftCentroidBBox = centroidBBox, rightCentroidBBox = centroidBBox;
        leftBBox[1][axis] = parentBBox[0][axis];
        rightBBox[0][axis] = parentBBox[1][axis];
        leftCentroidBBox[1][axis] = centroidBBox[0][axis];
        rightCentroidBBox[0][axis] = centroidBBox[1][axis];

        // Sort the primitive indices along the dominant axis
        int j = i0;
        uint sw;
        for (int i = i0; i < i1; ++i)
        {
            const BBox2f primBBox = m_getPrimitiveBBox(m_primitiveIdxs[i]);
            const vec2 centroid = primBBox.Centroid();

            if (centroid[axis] < split)
            {
                // Update the left partition position and the moment
                leftBBox[1][axis] = max(leftBBox[1][axis], primBBox[1][axis]);
                leftCentroidBBox[1][axis] = max(leftCentroidBBox[1][axis], centroid[axis]);
                // Swap the element into the left-hand partition
                sw = m_primitiveIdxs[j]; m_primitiveIdxs[j] = m_primitiveIdxs[i]; m_primitiveIdxs[i] = sw;
                // Increment the partition index
                ++j;
            }
            else
            {
                // Update the right partition position and centroid
                rightBBox[0][axis] = min(rightBBox[0][axis], primBBox[0][axis]);
                rightCentroidBBox[0][axis] = min(rightCentroidBBox[0][axis], centroid[axis]);
            }
        }

        // Grow the node vector by two
        const int leftIdx = m_hostNodes.Size();
        const int rightIdx = leftIdx + 1;
        m_hostNodes.Grow(2);   

        // Refresh the reference and build the inner node          
        m_hostNodes[thisIdx].MakeInner(leftIdx, axis, leftBBox[1][axis], rightBBox[0][axis], i0, i1);
        m_stats.numInnerNodes++;

        /*if (m_debugFunctor)
        {
            m_debugFunctor("----------\n");
            m_debugFunctor(tfm::format("Inner: %i (%i, %i -> %i)\n", thisIdx, leftIdx, axis, m_hostNodes[thisIdx].data).c_str());
            for (int idx = 0; idx < m_hostNodes.Size(); ++idx)
            {
                if (m_hostNodes[idx].IsLeaf())
                    m_debugFunctor(tfm::format("  %i -> Leaf %i\n", idx, m_hostNodes[idx].data.primIdxs[0]).c_str());
                else
                    m_debugFunctor(tfm::format("  %i -> Inner %i, %i\n", idx, m_hostNodes[idx].GetAxis(), m_hostNodes[idx].GetChildIndex()).c_str());
            }
            m_debugFunctor("\n");
        }*/

        // Build the child nodes
        BuildPartition(i0, j, leftIdx, depth + 1, leftBBox, leftCentroidBBox);
        BuildPartition(j, i1, rightIdx, depth + 1, rightBBox, rightCentroidBBox);
    }

    __host__ void Host::BIH2DAsset::Build(std::function<BBox2f(uint)>& functor)
    {
        Assert(m_hostNodes);
        //AssertMsg(!m_primitiveIdxs.empty(), "BIH builder does not contain any primitives.");

        // Resize and reset pointers
        BIH2DBuilder builder(*this, m_primitiveIdxs, 5, functor);
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

        Cuda::Synchronise(cu_deviceInstance, m_params);
    }

    __host__ void Host::BIH2DAsset::Synchronise()
    {
        //const BIH2DParams params = {m_isConstructed, m_treeBBox, m_hostNodes->GetDeviceInstance() };
        //Cuda::Synchronise(cu_deviceInstance, params);
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