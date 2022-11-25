#include "UIInspector.cuh"
#include "core/math/ColourUtils.cuh"
#include "../bih/BIH2DAsset.cuh"

namespace Enso
{
    __device__ vec4 Device::UIInspector::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {
        if (!m_worldBBox.Contains(pWorld)) { return vec4(0.f); }

        float distance = length2(pWorld - m_transform.trans);

        float outerRadius = 0.5f * m_worldBBox.EdgeLength(0) - viewCtx.dPdXY;
        if (distance > sqr(outerRadius)) { return vec4(0.f); }
        float innerRadius = 0.4 * m_worldBBox.EdgeLength(0) - viewCtx.dPdXY;
        if (distance < sqr(innerRadius)) { return vec4(0.f); }

        distance = sqrt(distance);
        return vec4(kOne, saturatef((outerRadius - distance) / viewCtx.dPdXY) * saturatef((distance - innerRadius) / viewCtx.dPdXY));
    }

    __host__ AssetHandle<Host::SceneObject> Host::UIInspector::Instantiate(const std::string& id)
    {
        return CreateAsset<Host::UIInspector>(id);
    }

    __host__ Host::UIInspector::UIInspector(const std::string& id) :
        Host::Tracable(id, m_hostInstance)
    {
        Log::Success("Host::UIInspector::UIInspector");

        cu_deviceInstance = InstantiateOnDevice<Device::UIInspector>();

        Synchronise(kSyncObjects);
    }

    __host__ Host::UIInspector::~UIInspector()
    {
        Log::Error("Host::UIInspector::~UIInspector");
        OnDestroyAsset();
    }

    __host__ void Host::UIInspector::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceInstance);
    }

    __host__ void Host::UIInspector::Synchronise(const int type)
    {
        Host::Tracable::Synchronise(cu_deviceInstance, type);

        if (type & kSyncParams) { SynchroniseInheritedClass<UIInspectorParams>(cu_deviceInstance, *this, kSyncParams); }
    }

    __host__ uint Host::UIInspector::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        //AssertInThread("kMainThread");

        Log::Warning(stateID);
        
        if (stateID == "kCreateSceneObjectOpen" || stateID == "kCreateSceneObjectHover")
        {
            // Set the position and bounding box of the widget
            SceneObjectParams::m_transform.trans = viewCtx.mousePos;
            SceneObjectParams::m_worldBBox = BBox2f(viewCtx.mousePos - vec2(viewCtx.dPdXY * 20.0f), viewCtx.mousePos + vec2(viewCtx.dPdXY * 20.0f));

            //SetDirtyFlags(kGI2DDirtyTransforms);
            SetDirtyFlags(kGI2DDirtyBVH);
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            Finalise();

            //SetDirtyFlags(kGI2DDirtyTransforms);
            SetDirtyFlags(kGI2DDirtyBVH);
        }        

        return m_dirtyFlags;
    }

    __host__ bool Host::UIInspector::IsConstructed() const
    {
        return true;
    }

    __host__ bool Host::UIInspector::Finalise()
    {
        m_isFinalised = true;
        return true;
    }

    __host__ bool Host::UIInspector::Rebuild(const uint parentFlags, const UIViewCtx& viewCtx)
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