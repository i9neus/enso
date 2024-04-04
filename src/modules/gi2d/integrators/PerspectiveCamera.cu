#include "PerspectiveCamera.cuh"

#include "../primitives/Ellipse.cuh"
#include "../primitives/GenericIntersector.cuh"
#include "io/json/JsonUtils.h"
#include "../primitives/SDF.cuh"

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
        vec4 L(0.f);

        L = vec4(kOne, SDF::Renderer::Line(pWorld, m_transform.trans, m_direction, 3.f, viewCtx.dPdXY));

        return L;
    }

    __host__ __device__ uint Device::PerspectiveCamera::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return kSceneObjectDelegatedAction;
    }

    __host__ __device__ void Device::PerspectiveCamera::OnSynchronise(const int syncFlags)
    {
        if (syncFlags == kSyncParams)
        {
            
        }
    }

    __host__ AssetHandle<Host::GenericObject> Host::PerspectiveCamera::Instantiate(const std::string& id, const Json::Node&)
    {
        return CreateAsset<Host::PerspectiveCamera>(id);
    }

    __host__ Host::PerspectiveCamera::PerspectiveCamera(const std::string& id) :
        Host::Camera2D(id, m_hostInstance)
    {
        cu_deviceInstance = InstantiateOnDevice<Device::PerspectiveCamera>();

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
    }

    __host__ void Host::PerspectiveCamera::Synchronise(const int syncFlags)
    {
        Camera2D::Synchronise(cu_deviceInstance, syncFlags);

        if (syncFlags & kSyncParams)
        {
            SynchroniseInheritedClass<PerspectiveCameraParams>(cu_deviceInstance, m_hostInstance, kSyncParams);
            m_hostInstance.OnSynchronise(syncFlags);
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
            m_hostInstance.m_transform.trans = viewCtx.mousePos;
            m_hostInstance.m_direction = vec2(0.f, 1.f);
        }
        else if (stateID == "kCreateSceneObjectHover")
        {
            if (m_onCreate.isCentroidSet)
            {
                m_hostInstance.m_direction = viewCtx.mousePos - m_hostInstance.m_transform.trans;
            }
            else
            {
                m_hostInstance.m_transform.trans = viewCtx.mousePos;
            }
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            if (!m_onCreate.isCentroidSet)
            {
                m_hostInstance.m_transform.trans = viewCtx.mousePos;
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
        return m_dirtyFlags;
    }

    __host__ bool Host::PerspectiveCamera::Rebuild(const uint parentFlags, const UIViewCtx& viewCtx)
    {
        //AssertInThread("kRenderThread");

        if (!m_dirtyFlags) { return IsConstructed(); }

        if (m_dirtyFlags & kDirtyObjectBounds)
        {
            RecomputeBoundingBoxes();
        }

        Synchronise(kSyncParams);
        ClearDirtyFlags();
        return IsConstructed();
    }

    __host__ bool Host::PerspectiveCamera::Serialise(Json::Node& node, const int flags) const
    {
        Camera2D::Serialise(node, flags);

        //node.AddValue("radius", m_hostInstance.m_lightRadius);
        //node.AddVector("colour", m_hostInstance.m_lightColour);
        //node.AddValue("intensity", m_hostInstance.m_lightIntensity);
        return true;
    }

    __host__ uint Host::PerspectiveCamera::Deserialise(const Json::Node& node, const int flags)
    {
        Camera2D::Deserialise(node, flags);

        //if (node.GetValue("radius", m_hostInstance.m_lightRadius, flags)) { SetDirtyFlags(kDirtyObjectBounds); }
        //if (node.GetVector("colour", m_hostInstance.m_lightColour, flags)) { SetDirtyFlags(kDirtyMaterials); }
        //if (node.GetValue("intensity", m_hostInstance.m_lightIntensity, flags)) { SetDirtyFlags(kDirtyMaterials); }

        return m_dirtyFlags;
    }

    __host__ uint Host::PerspectiveCamera::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return m_hostInstance.OnMouseClick(viewCtx);
    }

    __host__ BBox2f Host::PerspectiveCamera::RecomputeObjectSpaceBoundingBox()
    {
        BBox2f bBox;
        return Union(bBox, LineBBox2(vec2(0.f), m_hostInstance.m_direction));
    }
}
