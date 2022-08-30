#include "Curve.cuh"
#include "kernels/math/CudaColourUtils.cuh"

using namespace Cuda;

namespace GI2D
{    
    __host__ __device__ bool CurveInterface::IntersectRay(Ray2D& rayWorld, HitCtx2D& hitWorld) const
    {        
        assert(m_bih && m_lineSegments);
        
        RayRange2D range;
        if (!IntersectRayBBox(rayWorld, m_params.tracable.worldBBox, range) || range.tNear > hitWorld.tFar) { return false; }

        RayBasic2D& rayObject = ToObjectSpace(rayWorld);
        HitCtx2D hitObject;

        auto onIntersect = [&](const uint* startEndIdx, RayRange2D& rangeTree)
        {
            for (int primIdx = startEndIdx[0]; primIdx < startEndIdx[1]; ++primIdx)
            {
                if ((*m_lineSegments)[primIdx].IntersectRay(rayObject, hitObject) && hitObject.tFar < rangeTree.tFar && hitObject.tFar < hitWorld.tFar)
                {
                    rangeTree.tFar = hitObject.tFar;
                }
            }
        };
        m_bih->TestRay(rayObject, range.tFar, onIntersect);

        if (hitObject.tFar < hitWorld.tFar)
        {
            hitWorld.tFar = hitObject.tFar;
            return true;
        }

        return false;
    }

    /*__host__ __device__ bool CurveInterface::InteresectPoint(const vec2& p, const float& thickness) const
    {
    }*/

    /*__host__ __device__ vec2 CurveInterface::PerpendicularPoint(const vec2& p) const 
    {
    }*/

    __device__ vec4 CurveInterface::EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {
        vec4 L(0.0f);
        const vec2 pLocal = pWorld - m_params.tracable.transform.trans;

        m_bih->TestPoint(pLocal, [&, this](const uint* idxRange)
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    const auto& segment = (*m_lineSegments)[idx];
                    const float line = segment.Evaluate(pLocal, 0.001f, viewCtx.dPdXY);
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

    __device__ void Device::Curve::Synchronise(const CurveParams& params)
    {
        Super::Synchronise(params.tracable);
        m_params = params;
    }

    __host__ Host::Curve::Curve(const std::string& id) :
        Tracable(id),
        cu_deviceInstance(nullptr)
    {
        Log::Success("Host::Curve::Curve");
        
        constexpr uint kMinTreePrims = 3;
        
        m_hostBIH = CreateChildAsset<Host::BIH2DAsset>("bih", kMinTreePrims);
        m_hostLineSegments = CreateChildAsset<Cuda::Host::Vector<LineSegment>>("lineSegments", kVectorHostAlloc, nullptr);
        
        cu_deviceInstance = InstantiateOnDevice<Device::Curve>();
        cu_deviceTracableInterface = StaticCastOnDevice<TracableInterface>(cu_deviceInstance);

        m_deviceData.bih = m_hostBIH->GetDeviceInstance();
        m_deviceData.lineSegments = m_hostLineSegments->GetDeviceInstance();

        // Set the host parameters so we can query the primitive on the host
        m_bih = static_cast<BIH2D<BIH2DFullNode>*>(m_hostBIH.get());
        m_lineSegments = static_cast<Cuda::VectorInterface<GI2D::LineSegment>*>(m_hostLineSegments.get());

        SynchroniseObjects(cu_deviceInstance, m_deviceData);
        SynchroniseParams();
    }

    __host__ Host::Curve::~Curve()
    {
        Log::Error("Host::Curve::~Curve");
        OnDestroyAsset();
    }

    __host__ void Host::Curve::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceInstance);

        m_hostBIH.DestroyAsset();
        m_hostLineSegments.DestroyAsset();
    }

    __host__ void Host::Curve::SynchroniseParams()
    {
        m_params.tracable = Super::m_params;

        SynchroniseObjects(cu_deviceInstance, m_params);
    }

    __host__ uint Host::Curve::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        const vec2 mousePosLocal = viewCtx.mousePos - Super::m_params.transform.trans;
        if (stateID == "kCreateTracableOpen")
        {
            Super::m_params.transform.trans = viewCtx.mousePos;
           
            Log::Success("Opened path %s", GetAssetID());
        }
        else if (stateID == "kCreateTracableHover")
        {
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->Back().Set(1, mousePosLocal);
                SetDirtyFlags(kGI2DDirtyBVH);
            }
        }
        else if (stateID == "kCreateTracableAppend")
        {
            const vec3 colour = Hue(PseudoRNG(HashOf(m_hostLineSegments->Size())).Rand<0>());

            if (m_hostLineSegments->IsEmpty())
            {
                // Create a zero-length segment that will be manipulated later
                m_hostLineSegments->EmplaceBack(mousePosLocal, mousePosLocal, 0, colour);
            }
            else
            {
                // Any more and we simply reuse the last vertex on the path as the start of the next segment
                m_hostLineSegments->EmplaceBack(m_hostLineSegments->Back()[1], mousePosLocal, 0, colour);
            }

            SetDirtyFlags(kGI2DDirtyBVH);
        }
        else if (stateID == "kCreateTracableClose")
        {
            // Delete the floating segment when closing the path
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->PopBack();
            }

            Log::Warning("Closed path %s", GetAssetID());
            SetDirtyFlags(kGI2DDirtyBVH);
        }
        else
        {
            AssertMsg(false, "Invalid state");
        }

        return m_dirtyFlags;
    }

    __host__ bool Host::Curve::IsConstructed() const
    {
        return !m_hostLineSegments->IsEmpty() && m_hostBIH->IsConstructed();
    }

    __host__ bool Host::Curve::Finalise()
    {
        m_isFinalised = true;

        return IsConstructed();
    }

    __host__ bool Host::Curve::Rebuild(const uint parentFlags, const UIViewCtx& viewCtx)
    {
        if (!m_dirtyFlags) { return IsConstructed(); }

        bool resyncParams = false;
        
        // If the geometry has changed, rebuild the BIH
        if (m_dirtyFlags & kGI2DDirtyBVH)
        {
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

            // Update the tracable bounding boxes
            Super::m_params.objectBBox = m_hostBIH->GetBoundingBox();
            Super::m_params.worldBBox = Super::m_params.objectBBox + Super::m_params.transform.trans;
            //Log::Write("  - Rebuilt curve %s BIH: %s", GetAssetID(), GetObjectSpaceBoundingBox().Format()); 

            resyncParams = true;
        }

        if (m_dirtyFlags & kGI2DDirtyTransforms) 
        { 
            PrepareTransforms();
            resyncParams = true; 
        }

        if (resyncParams) { SynchroniseParams(); }

        ClearDirtyFlags();

        return IsConstructed();
    }
}