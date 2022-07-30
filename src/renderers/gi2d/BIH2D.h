#pragma once

#include "generic/StdIncludes.h"
#include "kernels/math/bbox/CudaBBox2.cuh"
#include <map>

enum BIHFlags : unsigned char 
{ 
    kBIHX = 0,
    kBIHY = 1,
    kBIHLeaf = 2
};

class BIH2D
{
public:
    struct Node
    {
        unsigned char   flags;
        union
        {
            int primIdx;
            struct
            {
                float           left;
                float           right;
            };
        };
    };

public:
    friend class BIH2DBuilder;

    BIH2D() : m_isConstructed(false) {}
    ~BIH2D() = default;

    bool IsConstructed() const { return m_isConstructed; }

    template<typename InteresectLambda>
    bool TestPoint(const Cuda::vec2& p, InteresectLambda onIntersect) const
    {
        TestPoint(p, 0, depth + 1, childBBox, onIntersect);
    }    

private:
    template<typename InteresectLambda>
    void TestPoint(const Cuda::vec2& p, const uint code, const uchar depth, const Cuda::BBox2f& parentBBox, InteresectLambda onIntersect) const
    {
        const uint nodeIdx = (1 << depth) - 1 + code;
        
        // Sanity checks
        Assert(nodeIdx < m_nodes.size());
        Assert(depth < 32);

        const Node& node = m_nodes[nodeIdx];
        if (node.flags == kBIHLeaf)
        {
            if (parentBBox.Contains(p))
            {
                onIntersect(node.primIdx);
            }
        }

        Cuda::BBox2f childBBox = parentBBox;
        childBBox[1][node.flags] = node.left;
        TestPoint(p, code << 1, depth + 1, childBBox, onIntersect);

        childBBox = parentBBox;
        childBBox[0][node.flags] = node.right;
        TestPoint(p, (code << 1) + 1, depth + 1, childBBox, onIntersect);
    }

private:
    std::vector<Node>           m_nodes;
    Cuda::BBox2f                m_bBox;
    bool                        m_isConstructed;
};

class BIH2DBuilder
{
public:
    BIH2DBuilder(BIH2D& bih) : m_bih(bih) {}

    template<typename GetBBoxLambda>
    void Build(std::vector<uint>& primitives, GetBBoxLambda getPrimitiveBBox, BIH2D& bih)
    {
        // Find the global bounding box
        m_bih.m_bBox = Cuda::BBox2f::MakeInvalid();

        AssertMsgFmt(m_bih.m_bBox.HasValidArea() && !m_bih.m_bBox.IsInfinite(),
            "BIH bounding box is invalid: {%s, %s}", m_bih.m_bBox[0].format().c_str(), m_bih.m_bBox[1].format().c_str());

        for (const auto idx : primitives)
        {
            const Cuda::BBox2f primBBox = getPrimitiveBBox(idx);
            AssertMsgFmt(primBBox.HasValidArea(), 
                        "BIH primitive at index %i has returned an invalid bounding box: {%s, %s}", primBBox[0].format().c_str(), primBBox[1].format().c_str());
            m_bih.m_bBox = Union(m_bih.bBox, primBBox);
        }

        // Construct the bounding interval hierarchy
        BuildPartition(primitives, 0, primitives.size(), 0, m_bih.m_bBox, getPrimitiveBBox);        
        
        m_bih.m_isConstructed = true;
    }

private:
    template<typename GetBBoxLambda>
    void BuildPartition(std::vector<uint>& primitives, const int i0, const int i1, const uint code, const uchar depth,
                        const Cuda::BBox2f& parentBBox, GetBBoxLambda getPrimitiveBBox)
    {
        Assert(depth < 32); // Sanity check
        
        if (i0 == i1) { return; }        
        
        const uint blockOffset = (1 << depth) - 1;
        Assert(blockOffset + code < m_bih.m_nodes.size()); // Sanity check

        // If this node only contains one primitive, it's a leaf
        if (i1 == i0 + 1)
        {
            BIH2D::Node& node = m_bih.m_nodes[blockOffset + code];
            node.flags = kBIHLeaf;
            node.primIdx = primitives[i0];
            return;
        }

        const int axis = parentBBox.MaxAxis();
        const float split = parentBBox.Centroid(axis);
        Cuda::BBox2f leftBBox = parentBBox, rightBBox = parentBBox;
        leftBBox[1][axis] = -kFltMax;
        rightBBox[0][axis] = kFltMax;

        // Sort the primitives along the dominant axis
        int j = 0;
        uint sw;
        for (int i = i0; i < i1; ++i)
        {
            const Cuda::BBox2f primBBox = getPrimitiveBBox(primitives[i]);
            const Cuda::vec2 centroid = primBBox.Centroid();

            if (centroid[axis] < split)
            {
                // Update the partition position
                leftPartition = max(leftPartition, leftBBox[1][axis]);
                // Swap the element into the left-hand partition
                sw = primitives[j]; primitives[j] = primitives[i]; primitives[i] = sw;
                // Increment the partition index
                ++j;
            }
            else
            {
                // Update the partition position
                rightPartition = min(rightPartition, rightBBox[0][axis]);
            }
        }
        
        // Build the inner node
        BIH2D::Node& node = m_bih.m_nodes[blockOffset + code];
        node.flags = axis;
        node.left = leftBBox[1][axis];
        node.right = rightBBox[0][axis];
        
        // Build the child nodes
        BuildPartition(primitives, i0, j, code << 1, depth + 1, leftBBox, getPrimitiveBBox);
        BuildPartition(primitives, j, i1, (code << 1) + 1, depth + 1, rightBBox, getPrimitiveBBox);
    }

private:
    BIH2D& m_bih;
};