#include "OmniLight.cuh"

#include "../primitives/Ellipse.cuh"
#include "../primitives/GenericIntersector.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __host__ __device__ vec4 Device::OmniLight::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        if (!GetWorldBBox().Contains(pWorld)) { return vec4(0.f); }

        return m_primitive.EvaluateOverlay(pWorld - GetTransform().trans, OverlayCtx::MakeStroke(viewCtx, vec4(1.), 3.f));
    }

    __host__ __device__ uint Device::OmniLight::OnMouseClick(const UIViewCtx& viewCtx) const
    {       
        return (m_primitive.Contains(viewCtx.mousePos - GetTransform().trans, viewCtx.dPdXY) > 0.0f) ? kSceneObjectPrecisionDrag : kSceneObjectInvalidSelect;
    }

    __host__ __device__ bool Device::OmniLight::IntersectRay(const Ray2D& rayWorld, HitCtx2D& hitWorld) const
    {
        RayRange2D range;
        if (!IntersectRayBBox(rayWorld, GetWorldBBox(), range) || range.tNear > hitWorld.tFar) { return false; }

        // TODO: Untransform normal
        return m_primitive.IntersectRay(RayBasic2D(rayWorld.o - GetTransform().trans, rayWorld.d), hitWorld);
    }

    __device__ bool Device::OmniLight::Sample(const Ray2D& parentRay, const HitCtx2D& hit, float xi, vec2& extant, vec3& L, float& pdf) const
    {
        const vec2 lightLocal = GetTransform().trans - hit.p;
        extant = normalize(vec2(lightLocal.y, -lightLocal.x)) * m_params.lightRadius * (xi - 0.5f) + lightLocal;
        float lightDist = length(extant);
        extant /= lightDist;

        float cosTheta = (hit.flags & kHit2DIsVolume) ? (1.0 / kTwoPi) : dot(extant, hit.n);
        if (cosTheta <= 0.0) { return false; }

        const float lightSolidAngle = 2.0f * ((lightDist >= m_params.lightRadius) ? asin(m_params.lightRadius / lightDist) : kHalfPi);

        L = m_params.lightColour * powf(2.0f, m_params.lightIntensity) * lightSolidAngle * cosTheta / (2.0 * m_params.lightRadius);
        pdf = 1.0 / lightSolidAngle;

        return true;
    }

    __device__ bool Device::OmniLight::Evaluate(const Ray2D& parentRay, const HitCtx2D& hit, vec3& L, float& pdfLight) const
    {

    }

    __device__ float Device::OmniLight::Estimate(const Ray2D& parentRay, const HitCtx2D& hit) const
    {
        return length(GetTransform().trans - hit.p) * powf(2.0f, m_params.lightIntensity);
    }

    __device__ void Device::OmniLight::OnSynchronise(const int syncFlags)
    {
        if (syncFlags == kSyncParams)
        {
            m_primitive = Ellipse(vec2(0.f), m_params.lightRadius);
        }
    }

    __host__ AssetHandle<Host::GenericObject> Host::OmniLight::Instantiate(const std::string& id, const Json::Node&, const AssetHandle<const Host::SceneDescription>&)
    {
        return CreateAsset<Host::OmniLight>(id);
    }

    __host__ Host::OmniLight::OmniLight(const std::string& id) :
        Host::Light(id, m_hostInstance),
        cu_deviceInstance(m_allocator.InstantiateOnDevice<Device::OmniLight>())
    {
        Light::SetDeviceInstance(m_allocator.StaticCastOnDevice<Device::Light>(cu_deviceInstance));

        Synchronise(kSyncObjects);
    }

    __host__ Host::OmniLight::~OmniLight()
    {
        BEGIN_EXCEPTION_FENCE

            OnDestroyAsset();

        END_EXCEPTION_FENCE
    }

    __host__ void Host::OmniLight::OnDestroyAsset()
    {
        m_allocator.DestroyOnDevice(cu_deviceInstance);
    }

    __host__ void Host::OmniLight::Synchronise(const uint syncFlags)
    {
        Light::Synchronise(syncFlags);

        if (syncFlags & kSyncParams) 
        { 
            SynchroniseObjects<>(cu_deviceInstance, m_hostInstance.m_params); 
            m_hostInstance.OnSynchronise(syncFlags);
        }
    }

    /*__host__ uint Host::OmniLight::OnMove(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        if (stateID != "kMoveSceneObjectDragging") { return 0; }

        GetTransform().trans = viewCtx.mousePos;
        SceneObject::m_params.worldBBox = BBox2f(GetTransform().trans - vec2(m_params.lightRadius), GetTransform().trans + vec2(m_params.lightRadius));

        SetDirtyFlags(kDirtyObjectBounds);
        return m_dirtyFlags;
    }*/

    __host__ uint Host::OmniLight::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        //AssertInThread("kMainThread");

        Log::Warning(stateID);

        if (stateID == "kCreateSceneObjectOpen")
        {
            // Set the origin of the 
            m_onCreate.isCentroidSet = false;
            m_isConstructed = true;
            m_hostInstance.GetTransform().trans = viewCtx.mousePos;
            m_hostInstance.m_params.lightRadius = viewCtx.dPdXY;
        }
        else if (stateID == "kCreateSceneObjectHover")
        {
            if (m_onCreate.isCentroidSet)
            {
                m_hostInstance.m_params.lightRadius = length(m_hostInstance.GetTransform().trans - viewCtx.mousePos);
            }
            else
            {
                m_hostInstance.GetTransform().trans = viewCtx.mousePos;
            }
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            if (!m_onCreate.isCentroidSet)
            {
                m_hostInstance.GetTransform().trans = viewCtx.mousePos;
                m_hostInstance.m_params.lightRadius = viewCtx.dPdXY;
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

    __host__ bool Host::OmniLight::Rebuild(const uint parentFlags, const UIViewCtx& viewCtx)
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

    __host__ bool Host::OmniLight::Serialise(Json::Node& node, const int flags) const
    {
        Tracable::Serialise(node, flags);

        node.AddValue("radius", m_hostInstance.m_params.lightRadius);
        node.AddVector("colour", m_hostInstance.m_params.lightColour);
        node.AddValue("intensity", m_hostInstance.m_params.lightIntensity);
        return true;
    }

    __host__ uint Host::OmniLight::Deserialise(const Json::Node& node, const int flags)
    {
        Tracable::Deserialise(node, flags);

        if (node.GetValue("radius", m_hostInstance.m_params.lightRadius, flags)) { SetDirtyFlags(kDirtyObjectBounds); }
        if (node.GetVector("colour", m_hostInstance.m_params.lightColour, flags)) { SetDirtyFlags(kDirtyMaterials); }
        if (node.GetValue("intensity", m_hostInstance.m_params.lightIntensity, flags)) { SetDirtyFlags(kDirtyMaterials); }

        return m_dirtyFlags;
    }

    __host__ uint Host::OmniLight::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return m_hostInstance.OnMouseClick(viewCtx);
    }

    __host__ BBox2f Host::OmniLight::RecomputeObjectSpaceBoundingBox()
    {
        return BBox2f(-vec2(m_hostInstance.m_params.lightRadius), vec2(m_hostInstance.m_params.lightRadius));
    }
}
