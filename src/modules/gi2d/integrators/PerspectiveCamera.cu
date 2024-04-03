#include "PerspectiveCamera.cuh"
#include "../tracables/primitives/LineSegment.cuh"
#include "../GenericIntersector.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __device__ vec4 Device::PerspectiveCamera::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {      
        return vec4(0.);
    }


    __device__ bool Device::PerspectiveCamera::CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const
    {
        return false;
    }

    __device__ void Device::PerspectiveCamera::Accumulate(const vec4& L, const RenderCtx& ctx)
    {
    }

    __host__ Host::PerspectiveCamera::PerspectiveCamera(const std::string& id) :
        Camera2D(id, m_hostInstance),
        cu_deviceInstance(nullptr)
    {
        SetAttributeFlags(kSceneObjectInteractiveElement);

        Log::Success("Host::PerspectiveCamera::PerspectiveCamera");     

        Synchronise(kSyncObjects);
    }

    __host__ AssetHandle<Host::GenericObject> Host::PerspectiveCamera::Instantiate(const std::string& id, const Json::Node&)
    {
        return CreateAsset<Host::PerspectiveCamera>(id);
    }

    __host__ Host::PerspectiveCamera::~PerspectiveCamera()
    {
        Log::Error("Host::PerspectiveCamera::~PerspectiveCamera");
        OnDestroyAsset();
    }

    __host__ void Host::PerspectiveCamera::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceInstance);
    }

    __host__ void Host::PerspectiveCamera::Synchronise(const int syncType)
    {
        Camera2D::Synchronise(cu_deviceInstance, syncType);

        if (syncType == kSyncObjects)
        {
            SynchroniseInheritedClass<PerspectiveCameraObjects>(cu_deviceInstance, m_deviceObjects, kSyncObjects);
        }
    }

    __host__ uint Host::PerspectiveCamera::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        //AssertInThread("kMainThread");

        Log::Warning(stateID);

        if (stateID == "kCreateSceneObjectOpen")
        {
            // Set the origin of the 
            m_onCreate.isCentroidSet = false;
            m_isConstructed = true;
            //m_hostInstance.m_lightRadius = viewCtx.dPdXY;
        }
        else if (stateID == "kCreateSceneObjectHover")
        {
            if (m_onCreate.isCentroidSet)
            {
                //m_hostInstance.m_lightRadius = length(m_hostInstance.m_transform.trans - viewCtx.mousePos);
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

    __host__ BBox2f Host::PerspectiveCamera::RecomputeObjectSpaceBoundingBox()
    {
        return BBox2f(-vec2(0.1), vec2(0.1));
    }

    __host__ bool Host::PerspectiveCamera::Serialise(Json::Node& node, const int flags) const
    {
        Camera2D::Serialise(node, flags);

        return true;
    }

    __host__ uint Host::PerspectiveCamera::Deserialise(const Json::Node& node, const int flags)
    {
        Camera2D::Deserialise(node, flags);

        return m_dirtyFlags;
    }

    __host__ bool Host::PerspectiveCamera::Contains(const UIViewCtx& viewCtx) const
    {
        return m_hostInstance.Contains(viewCtx);
    }
}