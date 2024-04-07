#include "PerspectiveCamera.cuh"

#include "../primitives/Ellipse.cuh"
#include "../primitives/LineSegment.cuh"
#include "../primitives/GenericIntersector.cuh"
#include "io/json/JsonUtils.h"
#include "../primitives/SDF.cuh"
#include "core/UIButtonMap.h"
#include "core/Vector.cuh"

namespace Enso
{
    __device__ bool Device::PerspectiveCamera::CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const
    {
        return false;
    }

    __device__ void Device::PerspectiveCamera::Accumulate(const vec4& L, const RenderCtx& ctx)
    {
    }
    
    __host__ __device__ vec4 Device::PerspectiveCamera::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {
        CudaAssertDebug(m_objects.ui.lineSegments);

        const vec2 pLocal = pWorld - GetTransform().trans;
        vec4 L(0.f);
        const OverlayCtx overlayCtx = OverlayCtx::MakeStroke(viewCtx, vec4(1.f), 3.f);
        for (int idx = 0; idx < m_objects.ui.lineSegments->Size(); ++idx)
        {
            L = Blend(L, (*m_objects.ui.lineSegments)[idx].EvaluateOverlay(pLocal, overlayCtx));
        }
        return L;
    }

    __host__ __device__ uint Device::PerspectiveCamera::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return kSceneObjectDelegatedAction;
    }

    __host__ AssetHandle<Host::GenericObject> Host::PerspectiveCamera::Instantiate(const std::string& id, const Json::Node&, const AssetHandle<const Host::SceneDescription>&)
    {
        return CreateAsset<Host::PerspectiveCamera>(id);
    }

    __host__ Host::PerspectiveCamera::PerspectiveCamera(const std::string& id) :
        Host::SceneObject(id, m_hostInstance, nullptr)
    {
        m_ui.hostLineSegments = CreateChildAsset<Host::Vector<LineSegment>>("uiLineSegments", kVectorHostAlloc);
        m_ui.hostEllipses = CreateChildAsset<Host::Vector<Ellipse>>("uiEllipses", kVectorHostAlloc);
        
        cu_deviceInstance = InstantiateOnDevice<Device::PerspectiveCamera>();        

        m_hostInstance.m_objects.ui.lineSegments = m_ui.hostLineSegments->GetDeviceInstance();
        m_hostInstance.m_objects.ui.ellipses = m_ui.hostEllipses->GetDeviceInstance();

        Synchronise(kSyncObjects);
    }

    __host__ Host::PerspectiveCamera::~PerspectiveCamera()
    {
        BEGIN_EXCEPTION_FENCE

            OnDestroyAsset();

        END_EXCEPTION_FENCE
    }

    __host__ void Host::PerspectiveCamera::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceInstance);

        m_ui.hostLineSegments.DestroyAsset();
        m_ui.hostEllipses.DestroyAsset();
    }

    __host__ void Host::PerspectiveCamera::Synchronise(const int syncFlags)
    {
        SceneObject::Synchronise(cu_deviceInstance, syncFlags);

        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<>(cu_deviceInstance, m_hostInstance.m_params); 
        }
        if (syncFlags & kSyncObjects)
        {        
            m_ui.hostLineSegments->Synchronise(kVectorSyncUpload);
            m_ui.hostEllipses->Synchronise(kVectorSyncUpload);

            SynchroniseObjects<>(cu_deviceInstance, m_hostInstance.m_objects);
        }
    }

    /*__host__ uint Host::PerspectiveCamera::OnMove(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        if (stateID != "kMoveSceneObjectDragging") { return 0; }

        m_transform.trans = viewCtx.mousePos;
        m_worldBBox = BBox2f(m_transform.trans - vec2(m_lightRadius), m_transform.trans + vec2(m_lightRadius));

        SetDirtyFlags(kDirtyObjectBounds);
        return m_dirtyFlags;
    }*/

    __host__ uint Host::PerspectiveCamera::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        //AssertInThread("kMainThread");

        Log::Warning(stateID);

        if (stateID == "kCreateSceneObjectOpen")
        {
            // Set the origin of the 
            m_onCreate.isCentroidSet = false;
            m_isConstructed = true;
            GetTransform().trans = viewCtx.mousePos;
            m_hostInstance.m_params.direction = vec2(0.f, 0.f);
        }
        else if (stateID == "kCreateSceneObjectHover")
        {
            if (m_onCreate.isCentroidSet)
            {
                m_hostInstance.m_params.direction = viewCtx.mousePos - GetTransform().trans;
            }
            else
            {
                GetTransform().trans = viewCtx.mousePos;
            }
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            if (!m_onCreate.isCentroidSet)
            {
                GetTransform().trans = viewCtx.mousePos;
                m_onCreate.isCentroidSet = true;
            }
            else
            {
                m_isFinalised = true;
            }
        }
        else
        {
            return m_dirtyFlags;
        }

        // If the object is dirty, recompute the bounding box
        SetDirtyFlags(kDirtyObjectBounds);
        UpdateUIElements();

        return m_dirtyFlags;
    }

    __host__ uint Host::PerspectiveCamera::OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx)
    {
        return m_dirtyFlags;
    }

    __host__ bool Host::PerspectiveCamera::Rebuild(const uint parentFlags, const UIViewCtx& viewCtx)
    {
        //AssertInThread("kRenderThread");

        if (!m_dirtyFlags) { return IsConstructed(); }

        if (m_dirtyFlags & kDirtyObjectBounds)
        {
            RecomputeBoundingBoxes();
            Synchronise(kSyncObjects);
        }

        Synchronise(kSyncParams);
        ClearDirtyFlags();
        return IsConstructed();
    }

    __host__ void Host::PerspectiveCamera::Render()
    {
    }

    __host__ bool Host::PerspectiveCamera::Serialise(Json::Node& node, const int flags) const
    {
        SceneObject::Serialise(node, flags);

        node.AddValue("fov", m_hostInstance.m_params.fov);
        return true;
    }

    __host__ uint Host::PerspectiveCamera::Deserialise(const Json::Node& node, const int flags)
    {
        SceneObject::Deserialise(node, flags);

        if (node.GetValue("fov", m_hostInstance.m_params.fov, flags)) { SetDirtyFlags(kDirtyObjectBounds); }

        return m_dirtyFlags;
    }

    __host__ uint Host::PerspectiveCamera::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return m_hostInstance.OnMouseClick(viewCtx);
    }

    __host__ BBox2f Host::PerspectiveCamera::RecomputeObjectSpaceBoundingBox()
    {
        // NOTE: UpdateUIElements() must be called before this method
        
        BBox2f bBox;
        for (const auto& segment : *m_ui.hostLineSegments)
        {
            bBox = Union(bBox, segment.GetBoundingBox());
        }
        return bBox;
    }

    __host__ void Host::PerspectiveCamera::UpdateUIElements()
    {
        // Create an orthonomal basis
        vec2 dir = normalize(m_hostInstance.m_params.direction) * 0.1f;
        mat2 basis(dir, vec2(-dir.y, dir.x));

        m_ui.hostLineSegments->Clear();
        m_ui.hostLineSegments->EmplaceBack(vec2(0.f), dir);
        m_ui.hostLineSegments->EmplaceBack(vec2(0.f), vec2(-dir.y, dir.x));

        // NOTE: Synchronise is called later when the scene is rebuilt
    }
}
