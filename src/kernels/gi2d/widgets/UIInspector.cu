#include "UIInspector.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "../BIH2DAsset.cuh"

using namespace Cuda;

namespace GI2D
{
    __device__ bool Device::UIInspector::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const
    {
        if (!m_params.sceneObject.worldBBox.Contains(pWorld)) { return false; }

        float distance = length2(pWorld - m_params.sceneObject.transform.trans);

        float outerRadius = 0.5f * m_params.sceneObject.worldBBox.EdgeLength(0) - viewCtx.dPdXY;
        if (distance > sqr(outerRadius)) { return false; }
        float innerRadius = 0.4 * m_params.sceneObject.worldBBox.EdgeLength(0) - viewCtx.dPdXY;
        if (distance < sqr(innerRadius)) { return false; }

        distance = sqrt(distance);
        L = Blend(L, kOne, saturatef((outerRadius - distance) / viewCtx.dPdXY) * saturatef((distance - innerRadius) / viewCtx.dPdXY));

        return true;
    }

    __device__ void Device::UIInspector::Synchronise(const Objects& objects)
    {
    
    }

    __device__ void Device::UIInspector::Synchronise(const UIInspectorParams& params)
    {
        SceneObjectInterface::Synchronise(params.sceneObject);
        m_params = params;

        //printf("{%f, %f}, {%f, %f}\n", 
        //    m_params.sceneObject.worldBBox.lower.x, m_params.sceneObject.worldBBox.lower.y,
        //    m_params.sceneObject.worldBBox.upper.x, m_params.sceneObject.worldBBox.upper.y);
    }

    __host__ AssetHandle<GI2D::Host::SceneObject> Host::UIInspector::Instantiate(const std::string& id)
    {
        return CreateAsset<GI2D::Host::UIInspector>(id);
    }

    __host__ Host::UIInspector::UIInspector(const std::string& id) :
        SceneObject(id),
        cu_deviceInstance(nullptr)
    {
        Log::Success("Host::UIInspector::UIInspector");

        cu_deviceInstance = InstantiateOnDevice<Device::UIInspector>();
        cu_deviceSceneObjectInterface = StaticCastOnDevice<SceneObjectInterface>(cu_deviceInstance);
        
        SynchroniseParams();
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

    __host__ void Host::UIInspector::SynchroniseParams()
    {
        SynchroniseObjects(cu_deviceInstance, m_params);
    }

    __host__ uint Host::UIInspector::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        //AssertInThread("kMainThread");

        Log::Warning(stateID);
        
        if (stateID == "kCreateSceneObjectOpen" || stateID == "kCreateSceneObjectHover")
        {
            // Set the position and bounding box of the widget
            m_params.sceneObject.transform.trans = viewCtx.mousePos;
            m_params.sceneObject.worldBBox = BBox2f(viewCtx.mousePos - vec2(viewCtx.dPdXY * 20.0f), viewCtx.mousePos + vec2(viewCtx.dPdXY * 20.0f));

            SetDirtyFlags(kGI2DDirtyTransforms);
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            Finalise();

            SetDirtyFlags(kGI2DDirtyTransforms);
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

        if (m_dirtyFlags & kGI2DDirtyTransforms)
        {
            resyncParams = true;
        }

        if (resyncParams) { SynchroniseParams(); }

        ClearDirtyFlags();

        return IsConstructed();
    }
}