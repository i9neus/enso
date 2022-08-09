#include "CudaBIH2D.cuh"

namespace Cuda
{
    __host__ __device__ BIH2D::BIH2D() :
        m_bBox(vec2(0.f), vec2(0.f)),
        m_nodes(nullptr),
        m_isConstructed(false)
    {

    }    

    __device__ void Device::BIH2DAsset::Synchronise(const BIH2DParams& params)
    {
        assert(params.nodes);
        m_nodes = params.nodes->Data();
        m_bBox = params.bBox;
        m_isConstructed = params.isConstructed;
    }
    
    __host__ Host::BIH2DAsset::BIH2DAsset(const std::string& id) : 
        Asset(id),
        cu_deviceInstance(nullptr)
    {
        cu_deviceInstance = InstantiateOnDevice<Device::BIH2DAsset>(GetAssetID());

        m_hostNodes = CreateChildAsset<Host::Vector<BIH2DNode>>(tfm::format("%s_nodes", id), this, kVectorHostAlloc, m_hostStream);
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

    __host__ Host::BIH2DBuilder::BIH2DBuilder(Host::BIH2DAsset& bih, std::vector<uint>& primitiveIdxs, std::function<BBox2f(uint)>& functor) noexcept :
        m_bih(bih),
        m_hostNodes(*bih.m_hostNodes),
        m_primitiveIdxs(primitiveIdxs),
        m_getPrimitiveBBox(functor),
        m_stats(bih.m_stats)
    {
    }

    //template<typename NodeContainer>
    __host__ void Host::BIH2DBuilder::Build()
    {
        Timer timer;

        // Find the global bounding box
        m_bih.m_bBox = BBox2f::MakeInvalid();

        AssertMsg(m_getPrimitiveBBox, "BIH builder does not have a valid bounding box functor.");

        BBox2f centroidBBox = BBox2f::MakeInvalid();
        for (const auto idx : m_primitiveIdxs)
        {
            const BBox2f primBBox = m_getPrimitiveBBox(idx);
            AssertMsgFmt(primBBox.HasValidArea(),
                "BIH primitive at index %i has returned an invalid bounding box: {%s, %s}", primBBox[0].format().c_str(), primBBox[1].format().c_str());
            m_bih.m_bBox = Union(m_bih.m_bBox, primBBox);
            centroidBBox = Union(centroidBBox, primBBox.Centroid());
        }        

        AssertMsgFmt(m_bih.m_bBox.HasValidArea() && !m_bih.m_bBox.IsInfinite(),
            "BIH bounding box is invalid: %s", m_bih.m_bBox.Format().c_str());

        // Reserve space for the nodes
        m_hostNodes.Reserve(m_primitiveIdxs.size());
        m_hostNodes.Resize(1);

        // Construct the bounding interval hierarchy
        m_stats = BIH2DStats();
        m_stats.numInnerNodes = 0;

        BuildPartition(0, m_primitiveIdxs.size(), 0, 0, m_bih.m_bBox, centroidBBox);

        // Update the host data structures
        m_bih.m_nodes = m_hostNodes.GetHostData();
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
                m_hostNodes[thisIdx].MakeLeaf(BIH2DNode::kInvalidLeaf);
                //return;
            }
            else if (i1 - i0 == 1)
            {
                m_stats.numLeafNodes++;
                m_hostNodes[thisIdx].MakeLeaf(m_primitiveIdxs[i0]);
                //return;
            }

            if (m_debugFunctor)
            {
                m_debugFunctor("----------\n");
                m_debugFunctor(tfm::format("Leaf: %i (%i)\n", thisIdx, m_hostNodes[thisIdx].data).c_str());
                for (int idx = 0; idx < m_hostNodes.Size(); ++idx)
                {
                    if(m_hostNodes[idx].IsLeaf())
                        m_debugFunctor(tfm::format("  %i -> Leaf %i\n", idx, m_hostNodes[idx].idx).c_str());
                    else
                        m_debugFunctor(tfm::format("  %i -> Inner %i, %i\n", idx, m_hostNodes[idx].GetAxis(), m_hostNodes[idx].GetChildIndex()).c_str());
                }
                m_debugFunctor("\n");
            }
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
        m_hostNodes[thisIdx].MakeInner(leftIdx, axis, leftBBox[1][axis], rightBBox[0][axis]);
        m_stats.numInnerNodes++;

        if (m_debugFunctor)
        {
            m_debugFunctor("----------\n");
            m_debugFunctor(tfm::format("Inner: %i (%i, %i -> %i)\n", thisIdx, leftIdx, axis, m_hostNodes[thisIdx].data).c_str());
            for (int idx = 0; idx < m_hostNodes.Size(); ++idx)
            {
                if (m_hostNodes[idx].IsLeaf())
                    m_debugFunctor(tfm::format("  %i -> Leaf %i\n", idx, m_hostNodes[idx].idx).c_str());
                else
                    m_debugFunctor(tfm::format("  %i -> Inner %i, %i\n", idx, m_hostNodes[idx].GetAxis(), m_hostNodes[idx].GetChildIndex()).c_str());
            }
            m_debugFunctor("\n");
        }

        // Build the child nodes
        BuildPartition(i0, j, leftIdx, depth + 1, leftBBox, leftCentroidBBox);
        BuildPartition(j, i1, rightIdx, depth + 1, rightBBox, rightCentroidBBox);

        auto temp = m_hostNodes[thisIdx];
    }

    __host__ void Host::BIH2DAsset::Build(std::function<BBox2f(uint)>& functor)
    {
        Assert(m_hostNodes);
        AssertMsg(!m_primitiveIdxs.empty(), "BIH builder does not contain any primitives.");

        // Resize and reset pointers
        BIH2DBuilder builder(*this, m_primitiveIdxs, functor);
        builder.m_debugFunctor = m_debugFunctor;
        builder.Build();

        // Synchronise the node data to the device
        m_hostNodes->Synchronise(kVectorSyncUpload);
        Cuda::Synchronise(cu_deviceInstance, BIH2DParams{ m_isConstructed, m_bBox, m_hostNodes->GetDeviceInstance() });
    }

    __host__ void Host::BIH2DAsset::Synchronise()
    {
        //const BIH2DParams params = {m_isConstructed, m_bBox, m_hostNodes->GetDeviceInstance() };
        //Cuda::Synchronise(cu_deviceInstance, params);
    }        
}