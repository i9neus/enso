#include "LineStrip.cuh"
#include "../primitives/LineSegment.cuh"
#include "../primitives/GenericIntersector.cuh"
#include "core/math/ColourUtils.cuh"
#include "../bih/BIH2DAsset.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __host__ __device__ bool Device::LineStrip::IntersectRay(const Ray2D& rayWorld, HitCtx2D& hitWorld) const
    {
        CudaAssertDebug(m_objects.bih && m_objects.lineSegments);

        RayRange2D range;
        if (!IntersectRayBBox(rayWorld, GetWorldBBox(), range) || range.tNear > hitWorld.tFar) { return false; }

        RayBasic2D& rayObject = RayToObjectSpace(rayWorld);
        HitCtx2D hitObject;

        auto onIntersect = [&](const uint* startEndIdx, RayRange2D& rangeTree)
        {
            for (int primIdx = startEndIdx[0]; primIdx < startEndIdx[1]; ++primIdx)
            {
                if ((*m_objects.lineSegments)[primIdx].IntersectRay(rayObject, hitObject) && hitObject.tFar < rangeTree.tFar && hitObject.tFar < hitWorld.tFar)
                {
                    rangeTree.tFar = hitObject.tFar;
                }
            }
        };
        m_objects.bih->TestRay(rayObject, range.tFar, onIntersect);

        /*for (int primIdx = 0; primIdx < m_objects.lineSegments->Size(); ++primIdx)
        {
            (*m_objects.lineSegments)[primIdx].IntersectRay(rayObject, hitObject);
        }*/

        if (hitObject.tFar < hitWorld.tFar)
        {
            hitWorld.tFar = hitObject.tFar;
            // TODO: Transform normals into screen space
            hitWorld.n = hitObject.n;
            hitWorld.kickoff = hitObject.kickoff;
            return true;
        }

        return false;
    }

    __host__ __device__ void Device::LineStrip::Print() const
    {
        printf("LineStrip::Print()\n");
        for (int idx = 0; idx < m_objects.lineSegments->Size(); ++idx)
        {
            const vec2& v0 = (*m_objects.lineSegments)[idx][0];
            const vec2& v1 = (*m_objects.lineSegments)[idx][1];
            printf("%i: [%f, %f] -> [%f, %f]\n", idx, v0.x, v0.y, v1.x, v1.y);
        }
    }

    /*__host__ __device__ bool Device::LineStrip::InteresectPoint(const vec2& p, const float& thickness) const
    {
    }*/

    /*__host__ __device__ vec2 Device::LineStrip::PerpendicularPoint(const vec2& p) const
    {
    }*/

    __host__ __device__ uint Device::LineStrip::OnMouseClick(const UIViewCtx& viewCtx) const
    {        
        return (EvaluateOverlay(viewCtx.mousePos, viewCtx, true).w > 0.f) ? kSceneObjectPrecisionDrag : kSceneObjectInvalidSelect;
    }

    __host__ __device__ vec4 Device::LineStrip::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        if (!m_objects.bih) { return vec4(0.0f); }

        const vec2 pLocal = pWorld - GetTransform().trans;
        vec4 L(0.0f);
        const OverlayCtx overlayCtx = OverlayCtx::MakeStroke(viewCtx, vec4(1.f), 3.f);

        m_objects.bih->TestPoint(pLocal, [&, this](const uint* idxRange) -> bool
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    const auto& segment = (*m_objects.lineSegments)[idx];
                    const vec4 line = segment.EvaluateOverlay(pLocal, overlayCtx);
                    if (line.w > 0.f) { L = Blend(L, line); }
                }
                return false;
            });

        return L;
    }

    __host__ Host::LineStrip::LineStrip(const Asset::InitCtx& initCtx) :
        Tracable(initCtx, m_hostInstance, nullptr),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::LineStrip>(*this))
    {
        SetAttributeFlags(kSceneObjectInteractiveElement);
        Tracable::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Tracable>(cu_deviceInstance));

        Log::Success("Host::LineStrip::LineStrip");

        constexpr uint kMinTreePrims = 3;

        m_hostBIH = AssetAllocator::CreateChildAsset<Host::BIH2DAsset>(*this, "bih", kMinTreePrims);
        m_hostLineSegments = AssetAllocator::CreateChildAsset<Host::Vector<LineSegment>>(*this, "lineSegments", kVectorHostAlloc);


        m_deviceObjects.bih = m_hostBIH->GetDeviceInstance();
        m_deviceObjects.lineSegments = m_hostLineSegments->GetDeviceInstance();

        // Set the host parameters so we can query the primitive on the host
        m_hostInstance.m_objects.bih = static_cast<BIH2D<BIH2DFullNode>*>(m_hostBIH.GetRawPtr());
        m_hostInstance.m_objects.lineSegments = static_cast<Vector<LineSegment>*>(m_hostLineSegments.GetRawPtr());

        Synchronise(kSyncObjects);
    }

    __host__ AssetHandle<Host::GenericObject> Host::LineStrip::Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::SceneContainer>& scene)
    {
        return AssetAllocator::CreateChildAsset<Host::LineStrip>(parentAsset, id);
    }

    __host__ Host::LineStrip::~LineStrip() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);

        m_hostBIH.DestroyAsset();
        m_hostLineSegments.DestroyAsset();
    }

    __host__ void Host::LineStrip::Synchronise(const uint syncType)
    {
        Tracable::Synchronise(syncType);

        if (syncType & kSyncObjects)
        {
            SynchroniseObjects<Device::LineStrip>(cu_deviceInstance, m_deviceObjects);
        }
    }

    __host__ bool Host::LineStrip::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        const vec2 mousePosLocal = viewCtx.mousePos - GetTransform().trans;
        if (stateID == "kCreateSceneObjectOpen")
        {
            GetTransform().trans = viewCtx.mousePos;

            Log::Success("Opened path %s", GetAssetID());
        }
        else if (stateID == "kCreateSceneObjectHover")
        {
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->Back().Set(1, mousePosLocal);
            }
            SignalDirty({ kDirtyObjectBoundingBox, kDirtyObjectRebuild });
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            const vec3 colour = Hue(PseudoRNG(HashOf(m_hostLineSegments->Size())).Rand<0>());

            if (m_hostLineSegments->IsEmpty())
            {
                // Create a zero-length segment that will be manipulated later
                m_hostLineSegments->PushBack(LineSegment(mousePosLocal, mousePosLocal));
            }
            else
            {
                // Any more and we simply reuse the last vertex on the path as the start of the next segment
                m_hostLineSegments->PushBack(LineSegment(m_hostLineSegments->Back()[1], mousePosLocal));
            }

            SignalDirty({ kDirtyObjectBoundingBox, kDirtyObjectRebuild });
        }
        else if (stateID == "kCreateSceneObjectClose")
        {
            // Delete the floating segment when closing the path
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->PopBack();
            }

            Log::Warning("Closed path %s", GetAssetID());
            m_isFinalised = true;
            SignalDirty({ kDirtyObjectBoundingBox, kDirtyObjectRebuild });
        }
        else
        {
            AssertMsg(false, "Invalid state");
        }

        return m_isFinalised;
    }

    __host__ uint Host::LineStrip::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return m_hostInstance.OnMouseClick(viewCtx);
    }

    __host__ bool Host::LineStrip::IsConstructed() const
    {
        return !m_hostLineSegments->IsEmpty() && m_hostBIH->IsConstructed();
    }

    __host__ bool Host::LineStrip::Rebuild()
    {
        if (!IsDirty(kDirtyObjectRebuild)) { return false; }
        
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

        //Log::Write("  - Rebuilt curve %s BIH: %s", GetAssetID(), GetObjectSpaceBoundingBox().Format()); 

        // Update the tracable bounding boxes
        RecomputeBoundingBoxes();
        Synchronise(kSyncObjects);

        return true;
    }

    __host__ BBox2f Host::LineStrip::RecomputeObjectSpaceBoundingBox()
    {
        return m_hostBIH->GetBoundingBox();
    }

    __host__ bool Host::LineStrip::Serialise(Json::Node& node, const int flags) const
    {
        Tracable::Serialise(node, flags);

        // Serialise the entire curve including its vertices
        if (flags & kSerialiseAll)
        {
            // Represent the vertices as a 2D vector [[v0.x, v0.y], [v1.x, v1.y], .... , [vn.x, vn.y]]
            const auto& segments = *m_hostLineSegments;
            std::vector<std::vector<float>> vertices(segments.Size() + 1);
            vertices[0] = { segments[0][0].x, segments[0][0].y };
            for (int idx = 0; idx < segments.Size(); ++idx)
            {
                vertices[idx + 1] = { segments[idx][1].x, segments[idx][1].y };
            }
            node.AddArray2D("v", vertices);
        }

        return true;
    }

    __host__ bool Host::LineStrip::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = Tracable::Deserialise(node, flags);

        // Deserialise the entire curve including its vertices
        if (flags & kSerialiseAll)
        {
            std::vector<std::vector<float>> vertices; 
            isDirty |= node.GetArray2DValues("v", vertices, flags);

            auto& segments = *m_hostLineSegments;
            segments.Resize(vertices.size() - 1);
            for (int idx = 0; idx < segments.Size(); ++idx)
            {
                segments[idx][0] = vec2(vertices[idx][0], vertices[idx][1]);
                segments[idx][1] = vec2(vertices[idx+1][0], vertices[idx+1][1]);
            }

            SignalDirty(kDirtyObjectBoundingBox);
            //SetDirtyFlags(kDirtyObjectBounds | kDirtyObjectBVH);
        }

        return isDirty;
    }
}