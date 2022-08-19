#include "Curve.cuh"
#include "kernels/math/CudaColourUtils.cuh"

using namespace Cuda;

namespace GI2D
{
    __device__ Device::Curve::Curve()
    {
    }
    
    __device__ void Device::Curve::Synchronise(const Objects& objects)
    {
        m_objects = objects;
    }

    __device__ bool Device::Curve::Intersect(Ray2D& ray, HitCtx2D& hit, float& tFarParent) const
    {
        const RayBasic2D localRay = ToObjectSpace(ray); 
        const Cuda::Device::Vector<LineSegment>& segments = *m_objects.lineSegments;

        auto onIntersect = [&](const uint& startIdx, const uint& endIdx, float& tFarChild) -> float
        {
            for (uint idx = startIdx; idx < endIdx; ++idx)
            {
                if (segments[idx].TestRay(ray, hit))
                {
                    if (hit.tFar < tFarChild && hit.tFar < tFarParent)
                    {
                        tFarChild = hit.tFar;
                    }
                }
            }
        };
        m_objects.bih->TestRay(ray, onIntersect);

        tFarParent = min(tFarParent, hit.tFar);
    }

    __host__ Host::Curve::Curve(const std::string& id) :
        Tracable(id),
        cu_deviceInstance(nullptr)
    {
        m_hostBIH = CreateChildAsset<Host::BIH2DAsset>(tfm::format("%s_bih", id), this);
        m_hostLineSegments = CreateChildAsset<Cuda::Host::Vector<LineSegment>>(tfm::format("%s_lineSegments", id), this, kVectorHostAlloc, nullptr);
        
        cu_deviceInstance = InstantiateOnDevice<Device::Curve>(GetAssetID());

        m_deviceData.bih = m_hostBIH->GetDeviceInstance();
        m_deviceData.lineSegments = m_hostLineSegments->GetDeviceInstance();

        Synchronise();
    }

    __host__ Host::Curve::~Curve()
    {
        OnDestroyAsset();
    }

    __host__ void Host::Curve::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceInstance);

        m_hostBIH.DestroyAsset();
        m_hostLineSegments.DestroyAsset();
    }

    __host__ void Host::Curve::Synchronise()
    {
        SynchroniseObjects(cu_deviceInstance, m_deviceData);
    }

    __host__ uint Host::Curve::OnCreate(const std::string& stateID, const vec2& mousePos)
    {
        if (stateID == "kCreatePathHover")
        {
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->Back().Set(1, mousePos);
                return kGI2DDirtyGeometry;
            }
        }
        else if (stateID == "kCreatePathAppend")
        {
            const vec3 colour = Hue(PseudoRNG(HashOf(m_hostLineSegments->Size())).Rand<0>());

            if (m_hostLineSegments->IsEmpty())
            {
                // Create a zero-length segment that will be manipulated later
                m_hostLineSegments->EmplaceBack(mousePos, mousePos, 0, colour);
                return kGI2DDirtyGeometry;
            }
            else
            {
                // Any more and we simply reuse the last vertex on the path as the start of the next segment
                m_hostLineSegments->EmplaceBack(m_hostLineSegments->Back()[1], mousePos, 0, colour);
                return kGI2DDirtyGeometry;
            }
        }
        else if (stateID == "kCreatePathClose")
        {
            // Delete the floating segment when closing the path
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->PopBack();
                return kGI2DDirtyGeometry;
            }
        }
        else
        {
            AssertMsg(false, "Invalid state");
        }
    }

    __host__ uint Host::Curve::OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx)
    {
        /*if (stateID == "kSelectPathDragging")        
        {
            const bool wasLassoing = selectCtx.isLassoing;

            if (!selectCtx.isLassoing)
            {
                // Deselect all the path segments
                for (auto& segment : *m_hostLineSegments) { segment.SetFlags(k2DPrimitiveSelected, false); }

                selectCtx.lassoBBox = BBox2f(mousePos, mousePos);
                selectCtx.isLassoing = true;
                m_numSelected = 0;

                return kGI2DDirtyPrimitiveAttributes;
            }

            selectCtx.lassoBBox = Grow(Rectify(selectCtx.mouseBBox), viewCtx.dPdXY * 2.);
            selectCtx.selectedBBox = BBox2f::MakeInvalid();

            //std::lock_guard <std::mutex> lock(m_resourceMutex);
            if (m_hostBIH->IsConstructed())
            {
                const uint lastNumSelected = m_numSelected;

                auto onIntersectPrim = [this](const uint& startIdx, const uint& endIdx, const bool isInnerNode)
                {
                    // Inner nodes are tested when the bounding box envelops them completely. Hence, there's no need to do a bbox checks.
                    if (isInnerNode)
                    {
                        for (int idx = startIdx; idx < endIdx; ++idx) { (*m_hostLineSegments)[idx].SetFlags(k2DPrimitiveSelected, true); }
                        m_numSelected += endIdx - startIdx;
                    }
                    else
                    {
                        for (int idx = startIdx; idx < endIdx; ++idx)
                        {
                            const bool isCaptured = lineSegments[idx].Intersects(selection.lassoBBox);
                            if (isCaptured)
                            {
                                selection.selectedBBox = Union(selection.selectedBBox, lineSegments[idx].GetBoundingBox());
                                ++selection.numSelected;
                            }
                            lineSegments[idx].SetFlags(k2DPrimitiveSelected, isCaptured);
                        }
                    }
                };
                m_objects.sceneBIH->TestBBox(selection.lassoBBox, onIntersectPrim);

                // Only if the number of selected primitives has changed
                if (lastNumSelected != selection.numSelected)
                {
                    if (selection.numSelected > 0 && !wasLassoing)
                    {
                        selection.isLassoing = false;
                        m_uiGraph.SetState("kMovePathBegin");
                    }

                    SetDirtyFlags(kGI2DDirtyPrimitiveAttributes);
                }
            }

            Log::Success("Selecting!");*/
        return 0;
    }
}