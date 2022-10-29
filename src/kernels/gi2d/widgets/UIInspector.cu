#include "UIInspector.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "../BIH2DAsset.cuh"

using namespace Cuda;

namespace GI2D
{
    __device__ bool Device::UIInspector::EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const
    {
        if (!m_worldBBox.Contains(pWorld)) { return false; }

        float distance = length2(pWorld - m_transform.trans);

        float outerRadius = 0.5f * m_worldBBox.EdgeLength(0) - viewCtx.dPdXY;
        if (distance > sqr(outerRadius)) { return false; }
        float innerRadius = 0.4 * m_worldBBox.EdgeLength(0) - viewCtx.dPdXY;
        if (distance < sqr(innerRadius)) { return false; }

        distance = sqrt(distance);
        L = Blend(L, kOne, saturatef((outerRadius - distance) / viewCtx.dPdXY) * saturatef((distance - innerRadius) / viewCtx.dPdXY));

        return true;
    }

    __host__ AssetHandle<GI2D::Host::SceneObject> Host::UIInspector::Instantiate(const std::string& id)
    {
        return CreateAsset<GI2D::Host::UIInspector>(id);
    }

    __host__ Host::UIInspector::UIInspector(const std::string& id) :
        Host::Tracable(id)
    {
        Log::Success("Host::UIInspector::UIInspector");

        cu_deviceUIInspectorInstance = InstantiateOnDevice<Device::UIInspector>();
        cu_deviceTracableInstance = StaticCastOnDevice<Device::Tracable>(cu_deviceUIInspectorInstance);
        
        Synchronise(kSyncObjects);
    }

    __host__ Host::UIInspector::~UIInspector()
    {
        Log::Error("Host::UIInspector::~UIInspector");
        OnDestroyAsset();
    }

    __host__ void Host::UIInspector::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceUIInspectorInstance);
    }

    __host__ void Host::UIInspector::Synchronise(const int type)
    {
        Host::Tracable::Synchronise(cu_deviceUIInspectorInstance, type);

        if (type == kSyncParams) { SynchroniseInheritedClass<UIInspectorParams>(cu_deviceUIInspectorInstance, *this); }
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