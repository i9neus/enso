#include "Light.cuh"

namespace GI2D
{
    __device__ vec4 Device::OmniLight::EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {
        if (!m_worldBBox.Contains(pWorld)) { return vec4(0.f); }

        float distance = length2(pWorld - m_transform.trans);

        float outerRadius = m_lightRadius - viewCtx.dPdXY;
        if (distance > sqr(outerRadius)) { return vec4(0.f); }
        float innerRadius = m_lightRadius - viewCtx.dPdXY * 3.0f;
        if (distance < sqr(innerRadius)) { return vec4(0.f); }

        distance = sqrt(distance);
        return vec4(kOne, saturatef((outerRadius - distance) / viewCtx.dPdXY) * saturatef((distance - innerRadius) / viewCtx.dPdXY));
    }

    __host__ AssetHandle<GI2D::Host::SceneObjectInterface> Host::OmniLight::Instantiate(const std::string& id)
    {
        return CreateAsset<GI2D::Host::OmniLight>(id);
    }

    __host__ Host::OmniLight::OmniLight(const std::string& id) :
        Super(id)
    {
        Log::Success("Host::OmniLight::OmniLight");

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
        Super::Synchronise(cu_deviceInstance, type);

        if (type & kSyncParams) { SynchroniseInheritedClass<OmniLightParams>(cu_deviceInstance, *this, kSyncParams); }
    }

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
                m_lightPosWorld = viewCtx.mousePos;
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
        SetDirtyFlags(kGI2DDirtyBVH);
        m_worldBBox = BBox2f(m_transform.trans - vec2(m_lightRadius), m_transform.trans + vec2(m_lightRadius));

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

        bool resyncParams = false;

        if (m_dirtyFlags & kGI2DDirtyBVH)
        {
            resyncParams = true;
        }

        if (resyncParams)
        {
            Synchronise(kSyncParams);
        }

        ClearDirtyFlags();

        return IsConstructed();
    }
}
