#include "Curve.cuh"
#include "primitives/LineSegment.cuh"
#include "../GenericIntersector.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "../BIH2DAsset.cuh"

using namespace Cuda;

namespace GI2D
{    
    __host__ __device__ bool Device::Curve::IntersectRay(Ray2D& rayWorld, HitCtx2D& hitWorld) const
    {        
        assert(m_bih && m_lineSegments);
        
        RayRange2D range;
        if (!IntersectRayBBox(rayWorld, m_worldBBox, range) || range.tNear > hitWorld.tFar) { return false; }

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

    /*__host__ __device__ bool Device::Curve::InteresectPoint(const vec2& p, const float& thickness) const
    {
    }*/

    /*__host__ __device__ vec2 Device::Curve::PerpendicularPoint(const vec2& p) const 
    {
    }*/

    __device__ vec4 Device::Curve::EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {
        if (!m_bih) { return vec4(0.0f); }

        const vec2 pLocal = pWorld - m_transform.trans;
        vec4 L(0.0f);
        m_bih->TestPoint(pLocal, [&, this](const uint* idxRange)
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    const auto& segment = (*m_lineSegments)[idx];
                    const float line = segment.Evaluate(pLocal, viewCtx.dPdXY);
                    if (line > 0.f)
                    {
                        L = Blend(L, segment.IsSelected() ? vec3(1.0f, 0.1f, 0.0f) : kOne, line);
                    }                 
                }                
            });
     
        return L;
    }

    __host__ Host::Curve::Curve(const std::string& id) :
        Super(id),
        cu_deviceInstance(nullptr)
    {
        SetAttributeFlags(kSceneObjectInteractiveElement);
        
        Log::Success("Host::Curve::Curve");
        
        constexpr uint kMinTreePrims = 3;
        
        m_hostBIH = CreateChildAsset<Host::BIH2DAsset>("bih", kMinTreePrims);
        m_hostLineSegments = CreateChildAsset<Core::Host::Vector<LineSegment>>("lineSegments", Core::kVectorHostAlloc);
        
        cu_deviceInstance = InstantiateOnDevice<Device::Curve>();
        cu_deviceTracableInstance = StaticCastOnDevice<Device::Tracable>(cu_deviceInstance);        

        m_deviceObjects.m_bih = m_hostBIH->GetDeviceInstance();
        m_deviceObjects.m_lineSegments = m_hostLineSegments->GetDeviceInstance();

        // Set the host parameters so we can query the primitive on the host
        m_bih = static_cast<BIH2D<BIH2DFullNode>*>(m_hostBIH.get());
        m_lineSegments = static_cast<Core::Vector<GI2D::LineSegment>*>(m_hostLineSegments.get());
        
        Synchronise(kSyncObjects);
    }

    __host__ AssetHandle<GI2D::Host::SceneObjectInterface> Host::Curve::Instantiate(const std::string& id)
    {
        return CreateAsset<GI2D::Host::Curve>(id);
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

    __host__ void Host::Curve::Synchronise(const int syncType)
    {
        Super::Synchronise(cu_deviceInstance, syncType);

        if (syncType == kSyncObjects)
        {
            SynchroniseInheritedClass<CurveObjects>(cu_deviceInstance, m_deviceObjects, kSyncObjects);
        }
    }

    __host__ uint Host::Curve::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        const vec2 mousePosLocal = viewCtx.mousePos - SceneObjectParams::m_transform.trans;
        if (stateID == "kCreateSceneObjectOpen")
        {
            SceneObjectParams::m_transform.trans = viewCtx.mousePos;
           
            Log::Success("Opened path %s", GetAssetID());
        }
        else if (stateID == "kCreateSceneObjectHover")
        {
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->Back().Set(1, mousePosLocal);
                SetDirtyFlags(kGI2DDirtyBVH);
            }
        }
        else if (stateID == "kCreateSceneObjectAppend")
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
        else if (stateID == "kCreateSceneObjectClose")
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
            // Sync the line segmentsb
            auto& segments = *m_hostLineSegments;
            segments.Synchronise(Core::kVectorSyncUpload);

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
            SceneObjectParams::m_objectBBox = m_hostBIH->GetBoundingBox();
            SceneObjectParams::m_worldBBox = SceneObjectParams::m_objectBBox + SceneObjectParams::m_transform.trans;
            //Log::Write("  - Rebuilt curve %s BIH: %s", GetAssetID(), GetObjectSpaceBoundingBox().Format()); 

            resyncParams = true;
        }

        if (m_dirtyFlags & kGI2DDirtyTransforms) 
        { 
            resyncParams = true;
        }

        if (resyncParams) { Synchronise(kSyncParams); }

        ClearDirtyFlags();

        return IsConstructed();
    }
}