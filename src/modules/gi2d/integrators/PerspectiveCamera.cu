#include "PerspectiveCamera.cuh"

#include "io/json/JsonUtils.h"
#include "../primitives/Ellipse.cuh"
#include "../primitives/LineSegment.cuh"
#include "../primitives/GenericIntersector.cuh"
#include "../primitives/SDF.cuh"
#include "core/UIButtonMap.h"
#include "core/Vector.cuh"
#include "AccumulationBuffer.cuh"
#include "../tracables/Tracable.cuh"
#include "../widgets/UIHandle.cuh"

namespace Enso
{        
    __device__ bool Device::PerspectiveCamera::CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const
    {
        const uint probeIdx = kKernelIdx / Camera::m_params.accum.subprobesPerProbe;

        // Transform from screen space to view space
        ray.o = m_params.cameraPos;

        // Construct the ray direction
        const float aaJitter = renderCtx.rng.Rand<0>();
        const float alpha = (aaJitter + float(probeIdx)) / float(Camera::m_params.accum.numProbes);
        ray.d.x = 1.f;
        ray.d.y = tanf(toRad(m_params.fov * 0.5f)) * (2.f * alpha - 1.0f);
        ray.d = m_params.invBasis * normalize(ray.d);

        /*if (renderCtx.IsDebug())
        {
            ray.o = vec2(0.0f);
            ray.d = normalize(UILayer::m_params.viewCtx.mousePos - ray.o);
        }*/

        ray.throughput = vec3(1.0f);
        ray.flags = 0;
        ray.lightIdx = kTracableNotALight;

        // Initialise the hit context
        hit.flags = kHit2DIsVolume;
        hit.p = ray.o;
        hit.tFar = kFltMax;
        hit.depth = 0;

        //(*m_objects.accumBuffer)[kKernelIdx] += vec3(ray.o, 0.f);

        return true;
    }

    __device__ void Device::PerspectiveCamera::Accumulate(const vec4& L, const RenderCtx& ctx)
    {
        (*m_objects.accumBuffer)[kKernelIdx] += L.xyz;
    }

    __host__ __device__ vec4 Device::PerspectiveCamera::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        CudaAssertDebug(m_objects.ui.lineSegments->Size() == 3);
        CudaAssertDebug(m_objects.ui.handles->Size() == 2);

        const vec2 pObject = pWorld - GetTransform().trans;
        vec4 L(0.f);

        // Only render the accumulator when we're on the device
#ifdef __CUDA_ARCH__
        const vec2 bBasis = m_params.fwdBasis * pObject;
        if (bBasis.x > 0.f)
        {
            const float delta = 0.5f * bBasis.y / (bBasis.x * tan(toRad(m_params.fov * 0.5))) + 0.5f;
            if (delta >= 0.f && delta < 1.f)
            {
                const int idx = int(delta * Camera::m_params.accum.numProbes);
                L = vec4(m_objects.accumBuffer->Evaluate(idx), 0.5f);
            }
        }
#endif
        
        // Render the wireframe
        OverlayCtx overlayCtx = OverlayCtx::MakeStroke(viewCtx, vec4(kOne, 0.3f), 0.5f, OverlayCtx::kStrokeHashed);
        EvaluateRange(*m_objects.ui.lineSegments, 0, 3, pObject, overlayCtx, L);

        // Render the control handles
        for (int idx = 0; idx < m_objects.ui.handles->Size(); ++idx)
        {
            L = Blend(L, (*m_objects.ui.handles)[idx].EvaluateOverlay(pObject, viewCtx));
        }

        return L;
    }

    __host__ AssetHandle<Host::GenericObject> Host::PerspectiveCamera::Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::SceneContainer>& scene)
    {
        return AssetAllocator::CreateChildAsset<Host::PerspectiveCamera>(parentAsset, id, scene);
    }

    __host__ Host::PerspectiveCamera::PerspectiveCamera(const Asset::InitCtx& initCtx, const AssetHandle<const Host::SceneContainer>& scene) :
        Host::Camera(initCtx, &m_hostInstance, scene)
    {
        cu_deviceInstance = AssetAllocator::InstantiateOnDevice<Device::PerspectiveCamera>(*this);
        Camera::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Camera>(cu_deviceInstance));
        
        m_ui.hostLineSegments = AssetAllocator::CreateChildAsset<Host::Vector<LineSegment>>(*this, "uiLineSegments", kVectorHostAlloc);
        m_ui.hostUIHandles = AssetAllocator::CreateChildAsset<Host::Vector<UIHandle>>(*this, "uiHandles", kVectorHostAlloc);

        constexpr uint kGridWidth = 100;
        constexpr uint kGridHeight = 1;
        constexpr uint kNumHarmonics = 1;
        constexpr size_t kAccumBufferSize = 1000;

        // Construct the camera
        Camera::Initialise(kGridWidth * kGridHeight, kNumHarmonics, kAccumBufferSize);

        // Set device object pointers
        m_deviceObjects.accumBuffer = m_accumBuffer->GetDeviceInstance();
        m_deviceObjects.ui.lineSegments = m_ui.hostLineSegments->GetDeviceInstance();
        m_deviceObjects.ui.handles = m_ui.hostUIHandles->GetDeviceInstance();

        // Set the host object pointers
        m_hostInstance.m_objects.ui.lineSegments = &(*m_ui.hostLineSegments);
        m_hostInstance.m_objects.ui.handles = &(*m_ui.hostUIHandles);

        // Construct UI elements
        m_ui.hostLineSegments->Resize(3);
        m_ui.hostUIHandles->Resize(2);

        ConstructUIHandlesFromAxis(vec2(0.f, 0.f));
        ConstructUIWireframes();

        Synchronise(kSyncObjects | kSyncParams);
    }

    __host__ Host::PerspectiveCamera::~PerspectiveCamera() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);

        m_ui.hostLineSegments.DestroyAsset();
        m_ui.hostUIHandles.DestroyAsset();
    }

    __host__ void Host::PerspectiveCamera::Synchronise(const uint syncFlags)
    {
        SceneObject::Synchronise(syncFlags);

        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<>(cu_deviceInstance, m_hostInstance.m_params);
        }
        if (syncFlags & kSyncObjects)
        {
            m_ui.hostLineSegments->Synchronise(kVectorSyncUpload);
            m_ui.hostUIHandles->Synchronise(kVectorSyncUpload);

            SynchroniseObjects<>(cu_deviceInstance, m_deviceObjects);
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

    __host__ bool Host::PerspectiveCamera::OnCreateSceneObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject)
    {
        //AssertInThread("kMainThread");

        if (stateID == "kCreateSceneObjectOpen")
        {
            // Set the origin of the 
            m_onCreate.isCentroidSet = false;
            m_isConstructed = true;
            m_hostInstance.m_params.cameraAxis = vec2(0.f, 0.f);
        }
        else if (stateID == "kCreateSceneObjectHover")
        {
            if (m_onCreate.isCentroidSet)
            {
                m_hostInstance.m_params.cameraAxis = SafeNormalize(mousePosObject);
            }
            else
            {
                SetTransform(viewCtx.mousePos);
            }
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            if (!m_onCreate.isCentroidSet)
            {
                SetTransform(viewCtx.mousePos);
                m_onCreate.isCentroidSet = true;
            }
            else
            {
                m_isFinalised = true;
            }
        }
        else
        {
            return true;
        }

        // If the object is dirty, recompute the bounding box
        ConstructUIHandlesFromAxis(m_hostInstance.m_params.cameraAxis);
        ConstructUIWireframes();
        UpdateObjectSpaceBoundingBox();

        return true;
    }

    __host__ bool Host::PerspectiveCamera::OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx)
    {
        const vec2 mouseLocal = viewCtx.mousePos - GetTransform().trans;

        // Render the control handles
        if (GetAxisHandle().OnDelegateAction(stateID, keyMap, mouseLocal))
        {
            SignalDirty(kDirtyObjectRebuild);
        }
              
        return true;
    }

    __host__ bool Host::PerspectiveCamera::Rebuild()
    {
        SceneObject::Rebuild();
        
        // Create an orthonomal basis
        const vec2 dir = SafeNormalize(GetAxisHandle().GetCentroid());
        m_hostInstance.m_params.fwdBasis = mat2(dir, vec2(-dir.y, dir.x));
        m_hostInstance.m_params.invBasis = transpose(m_hostInstance.m_params.fwdBasis);
        m_hostInstance.m_params.cameraPos = GetTransform().trans;

        ConstructUIWireframes();
        UpdateObjectSpaceBoundingBox();
        Synchronise(kSyncObjects | kSyncParams);

        return true;
    }

    __host__ bool Host::PerspectiveCamera::Serialise(Json::Node& node, const int flags) const
    {
        SceneObject::Serialise(node, flags);
        Camera::Serialise(node, flags);

        node.AddValue("fov", m_hostInstance.m_params.fov);
        return true;
    }

    __host__ bool Host::PerspectiveCamera::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = Camera::Deserialise(node, flags);

        isDirty |= node.GetValue("fov", m_hostInstance.m_params.fov, flags);

        return isDirty;
    }

    __host__ uint Host::PerspectiveCamera::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        Assert(m_hostInstance.m_objects.ui.handles->Size() == 2);
        const vec2 pObject = viewCtx.mousePos - GetTransform().trans;

        if ((*m_hostInstance.m_objects.ui.handles)[0].EvaluateOverlay(pObject, viewCtx).w > 0.f)
        {
            return kSceneObjectPrecisionDrag;
        }
        else if ((*m_hostInstance.m_objects.ui.handles)[1].EvaluateOverlay(pObject, viewCtx).w != 0.f)
        {
            return kSceneObjectDelegatedAction;
        }

        return kSceneObjectInvalidSelect;
    }

    __host__ void Host::PerspectiveCamera::UpdateObjectSpaceBoundingBox()
    {
        // Caches the object space bounding box
        m_objectSpaceBBox.MakeInvalid();
        for (const auto& segment : *m_ui.hostLineSegments) { m_objectSpaceBBox = Union(m_objectSpaceBBox, segment.GetBoundingBox()); }
        for (const auto& control : *m_ui.hostUIHandles) { m_objectSpaceBBox = Union(m_objectSpaceBBox, control.GetBoundingBox()); }
    }

    __host__ void Host::PerspectiveCamera::ConstructUIWireframes()
    {
        constexpr float kFrustumExtent = 0.5f;
        constexpr float kViewPlane = 0.1f;

        const float theta = toRad(m_hostInstance.m_params.fov * 0.5);
        const vec2 frustum = vec2(cos(theta), sin(theta)) * kFrustumExtent;
        const vec2 viewPlane = vec2(1.f, tan(theta)) * kViewPlane;

        (*m_ui.hostLineSegments)[0] = LineSegment(vec2(0.f), m_hostInstance.m_params.invBasis * frustum);
        (*m_ui.hostLineSegments)[1] = LineSegment(vec2(0.f), m_hostInstance.m_params.invBasis * vec2(frustum.x, -frustum.y));
        (*m_ui.hostLineSegments)[2] = LineSegment(m_hostInstance.m_params.invBasis * viewPlane, m_hostInstance.m_params.invBasis * vec2(viewPlane.x, -viewPlane.y));

        // NOTE: Synchronise is called later when the scene is rebuilt
    }

    __host__ void Host::PerspectiveCamera::ConstructUIHandlesFromAxis(const vec2& cameraAxis)
    {
        constexpr float kHandleRadius = 0.005f;
        constexpr float kViewPlane = 0.1f;

        (*m_ui.hostUIHandles)[0] = UIHandle(vec2(0.f), kHandleRadius);
        (*m_ui.hostUIHandles)[1] = UIHandle(cameraAxis * kViewPlane, kHandleRadius);
    }

    __host__ UIHandle& Host::PerspectiveCamera::GetOriginHandle() { Assert(m_ui.hostUIHandles->Size() == 3); return (*m_ui.hostUIHandles)[0]; }
    __host__ UIHandle& Host::PerspectiveCamera::GetAxisHandle() { Assert(m_ui.hostUIHandles->Size() == 2); return (*m_ui.hostUIHandles)[1]; }

}
