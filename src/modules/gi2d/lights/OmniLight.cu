#include "OmniLight.cuh"

#include "../tracables/primitives/Ellipse.cuh"
#include "../GenericIntersector.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __device__ vec4 Device::OmniLight::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {
        if (!m_worldBBox.Contains(pWorld)) { return vec4(0.f); }

        return vec4(kOne, m_primitive.EvaluateOverlay(pWorld - m_transform.trans, viewCtx.dPdXY));
    }

    __host__ __device__ bool Device::OmniLight::IntersectRay(const Ray2D& rayWorld, HitCtx2D& hitWorld) const
    {
        RayRange2D range;
        if (!IntersectRayBBox(rayWorld, m_worldBBox, range) || range.tNear > hitWorld.tFar) { return false; }

        // TODO: Untransform normal
        return m_primitive.IntersectRay(RayBasic2D(rayWorld.o - m_transform.trans, rayWorld.d), hitWorld);
    }

    __device__ bool Device::OmniLight::Sample(const Ray2D& parentRay, const HitCtx2D& hit, float xi, vec2& extant, vec3& L, float& pdf) const
    {
        const vec2 lightLocal = m_transform.trans - hit.p;
        extant = normalize(vec2(lightLocal.y, -lightLocal.x)) * m_lightRadius * (xi - 0.5f) + lightLocal;
        float lightDist = length(extant);
        extant /= lightDist;

        float cosTheta = (hit.flags & kHit2DIsVolume) ? (1.0 / kTwoPi) : dot(extant, hit.n);
        if (cosTheta <= 0.0) { return false; }

        const float lightSolidAngle = 2.0f * ((lightDist >= m_lightRadius) ? asin(m_lightRadius / lightDist) : kHalfPi);

        L = m_lightColour * powf(2.0f, m_lightIntensity) * lightSolidAngle * cosTheta / (2.0 * m_lightRadius);
        pdf = 1.0 / lightSolidAngle;

        return true;
    }

    __device__ bool Device::OmniLight::Evaluate(const Ray2D& parentRay, const HitCtx2D& hit, vec3& L, float& pdfLight) const
    {

    }

    __device__ float Device::OmniLight::Estimate(const Ray2D& parentRay, const HitCtx2D& hit) const
    {
        return length(m_transform.trans - hit.p) * powf(2.0f, m_lightIntensity);
    }

    __device__ void Device::OmniLight::OnSynchronise(const int syncFlags)
    {
        if (syncFlags == kSyncParams)
        {
            m_primitive = Ellipse(vec2(0.f), m_lightRadius);
        }
    }

    __host__ AssetHandle<Host::GenericObject> Host::OmniLight::Instantiate(const std::string& id, const Json::Node&)
    {
        return CreateAsset<Host::OmniLight>(id);
    }

    __host__ Host::OmniLight::OmniLight(const std::string& id) :
        Host::Light(id, m_hostInstance)
    {
        cu_deviceInstance = InstantiateOnDevice<Device::OmniLight>();

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
        DestroyOnDevice(cu_deviceInstance);
    }

    __host__ void Host::OmniLight::Synchronise(const int type)
    {
        Light::Synchronise(cu_deviceInstance, type);

        if (type & kSyncParams) { SynchroniseInheritedClass<OmniLightParams>(cu_deviceInstance, *this, kSyncParams); }
    }

    /*__host__ uint Host::OmniLight::OnMove(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        if (stateID != "kMoveSceneObjectDragging") { return 0; }

        m_transform.trans = viewCtx.mousePos;
        m_worldBBox = BBox2f(m_transform.trans - vec2(m_lightRadius), m_transform.trans + vec2(m_lightRadius));

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
            m_lightRadius = viewCtx.dPdXY;
        }
        else if (stateID == "kCreateSceneObjectHover")
        {
            if (m_onCreate.isCentroidSet)
            {
                m_lightRadius = length(m_transform.trans - viewCtx.mousePos);
            }
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            if (!m_onCreate.isCentroidSet)
            {
                m_transform.trans = viewCtx.mousePos;
                m_lightRadius = viewCtx.dPdXY;
                m_onCreate.isCentroidSet = true;
            }
            else
            {
                Finalise();
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

    __host__ bool Host::OmniLight::IsConstructed() const
    {
        return true;
    }

    __host__ bool Host::OmniLight::Finalise()
    {
        m_isFinalised = true;
        return true;
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

        node.AddValue("radius", m_lightRadius);
        node.AddVector("colour", m_lightColour);
        node.AddValue("intensity", m_lightIntensity);
        return true;
    }

    __host__ uint Host::OmniLight::Deserialise(const Json::Node& node, const int flags)
    {
        Tracable::Deserialise(node, flags);

        if (node.GetValue("radius", m_lightRadius, flags)) { SetDirtyFlags(kDirtyObjectBounds); }
        if (node.GetVector("colour", m_lightColour, flags)) { SetDirtyFlags(kDirtyMaterials); }
        if (node.GetValue("intensity", m_lightIntensity, flags)) { SetDirtyFlags(kDirtyMaterials); }

        return m_dirtyFlags;
    }

    __host__ BBox2f Host::OmniLight::RecomputeObjectSpaceBoundingBox()
    {
        return BBox2f(-vec2(m_lightRadius), vec2(m_lightRadius));
    }
}
