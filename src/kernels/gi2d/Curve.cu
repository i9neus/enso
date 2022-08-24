#include "Curve.cuh"
#include "kernels/math/CudaColourUtils.cuh"

using namespace Cuda;

namespace GI2D
{    
    /*__host__ __device__ bool CurveInterface::IntersectRay(Ray2D& ray, HitCtx2D& hit, float& tFarParent) const
    {
        vec2 tNearFar;
        if (!IntersectRayBBox(ray, m_tracableBBox, tNearFar) || tNearFar[0] > tFarParent) { return false; }
        
        const RayBasic2D localRay = ToObjectSpace(ray);

        auto onIntersect = [&](const uint* startEndIdx, float& tFarChild)
        {
            for (int primIdx = startEndIdx[0]; primIdx < startEndIdx[1]; ++primIdx)
            {
                if ((*m_lineSegments)[primIdx].IntersectRay(ray, hit) && hit.tFar < tFarChild && hit.tFar < tFarParent)
                {
                    tFarChild = hit.tFar;
                }
            }
        };
        m_bih->TestRay(ray, onIntersect);

        tFarParent = min(tFarParent, hit.tFar);
        return 0;
    }

    __host__ __device__ bool CurveInterface::InteresectPoint(const vec2& p, const float& thickness) const
    {
    }

    __host__ __device__ bool CurveInterface::IntersectBBox(const BBox2f& bBox) const
    {
        return bBox.Intersects(m_tracableBBox);
    }

    __host__ __device__ vec2 CurveInterface::PerpendicularPoint(const vec2& p) const 
    {
    }*/

    __host__ __device__ vec4 CurveInterface::EvaluateOverlay(const vec2& p, const ViewTransform2D& viewCtx) const
    {
        vec4 L(0.0f);        
        m_bih->TestPoint(p, [&, this](const uint* idxRange)
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    const auto& segment = (*m_lineSegments)[idx];
                    const float line = segment.Evaluate(p, 0.001f, viewCtx.dPdXY);
                    if (line > 0.f)
                    {
                        L = Blend(L, segment.IsSelected() ? vec3(1.0f, 0.1f, 0.0f) : kOne, line);
                    }
                }
            });

        return L;
    }
    
    __device__ void Device::Curve::Synchronise(const Objects& objects)
    {
        m_bih = objects.bih;
        m_lineSegments = objects.lineSegments;
    }

    __host__ Host::Curve::Curve(const std::string& id) :
        Tracable(id),
        cu_deviceInstance(nullptr)
    {
        constexpr uint kMinTreePrims = 3;
        
        m_hostBIH = CreateChildAsset<Host::BIH2DAsset>("bih", kMinTreePrims);
        m_hostLineSegments = CreateChildAsset<Cuda::Host::Vector<LineSegment>>("lineSegments", kVectorHostAlloc, nullptr);
        
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

    __host__ uint Host::Curve::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        if (stateID == "kCreatePathOpen")
        {
            Log::Success("Opened path %s", GetAssetID());
        }
        else if (stateID == "kCreatePathHover")
        {
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->Back().Set(1, viewCtx.mousePos);
                SetDirtyFlags(kGI2DDirtyGeometry);
            }
        }
        else if (stateID == "kCreatePathAppend")
        {
            const vec3 colour = Hue(PseudoRNG(HashOf(m_hostLineSegments->Size())).Rand<0>());

            if (m_hostLineSegments->IsEmpty())
            {
                // Create a zero-length segment that will be manipulated later
                m_hostLineSegments->EmplaceBack(viewCtx.mousePos, viewCtx.mousePos, 0, colour);
                SetDirtyFlags(kGI2DDirtyGeometry);
            }
            else
            {
                // Any more and we simply reuse the last vertex on the path as the start of the next segment
                m_hostLineSegments->EmplaceBack(m_hostLineSegments->Back()[1], viewCtx.mousePos, 0, colour);
                SetDirtyFlags(kGI2DDirtyGeometry);
            }
        }
        else if (stateID == "kCreatePathClose")
        {
            // Delete the floating segment when closing the path
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->PopBack();
            }

            Log::Warning("Closed path %s", GetAssetID());
            SetDirtyFlags(kGI2DDirtyGeometry);
        }
        else
        {
            AssertMsg(false, "Invalid state");
        }

        return m_dirtyFlags;
    }

    __host__ bool Host::Curve::IsEmpty() const
    {
        return m_hostLineSegments->IsEmpty();
    }

    __host__ bool Host::Curve::Finalise()
    {
        m_isFinalised = true;
        return !m_hostLineSegments->IsEmpty();
    }

    __host__ void Host::Curve::Rebuild()
    {
        if (!(m_dirtyFlags & kGI2DDirtyGeometry)) { return; }
        
        // Sync the line segments
        auto& segments = *m_hostLineSegments;
        segments.Synchronise(kVectorSyncUpload);

        // Create a segment list ready for building
        // TODO: It's probably faster if we build on the already-sorted index list
        auto& primIdxs = m_hostBIH->GetPrimitiveIndices();
        primIdxs.resize(segments.Size());
        for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

        // Construct the BIH
        std::function<BBox2f(uint)> getPrimitiveBBox = [&segments](const uint& idx) -> BBox2f
        {
            return Grow(segments[idx].GetBoundingBox(), 0.001f);
        };
        m_hostBIH->Build(getPrimitiveBBox);

        // Update the tracable bounding box 
        m_tracableBBox = m_hostBIH->GetBoundingBox();

        Log::Write("  - Rebuilt curve %s BIH: %s", GetAssetID(), m_tracableBBox.Format());

        ClearDirtyFlags(kGI2DDirtyAll);

    //SetDirtyFlags(kGI2DDirtyPrimitiveAttributes);
    }

    /*__host__ uint Host::Curve::OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx)
    {
        if (stateID == "kSelectPathDragging")        
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

            Log::Success("Selecting!");
        return 0;
    }*/
}