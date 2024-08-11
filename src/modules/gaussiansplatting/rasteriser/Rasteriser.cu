#include "Rasteriser.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/3d/Ctx.cuh"
#include "core/AssetContainer.cuh"
#include "core/3d/Transform.cuh"
#include "core/Vector.cuh"
#include "core/3d/Cameras.cuh"
#include "core/GenericObjectContainer.cuh"

#include "../scene/SceneContainer.cuh"
#include "../scene/cameras/Camera.cuh"
#include "../scene/materials/Material.cuh"
#include "../scene/lights/Light.cuh"
#include "../scene/textures/Texture2D.cuh"
#include "../scene/tracables/Tracable.cuh"

#include "io/json/JsonUtils.h"
//#include "core/AccumulationBuffer.cuh"

namespace Enso
{        
    __host__ __device__ RasteriserParams::RasteriserParams()
    {
        viewport.dims = ivec2(0);
        frameIdx = 0;
        hasValidScene = false;
    }

    __host__ __device__ void RasteriserParams::Validate() const
    {
        CudaAssert(viewport.dims.x != 0 && viewport.dims.y != 0);
    }

    __host__ __device__ void Device::Rasteriser::Synchronise(const RasteriserParams& params) 
    {
        m_params = params;   
     }
    
    __device__ void Device::Rasteriser::Synchronise(const RasteriserObjects& objects) 
    { 
        m_objects = objects;

    }

    __device__ void Device::Rasteriser::Render()
    {
        

    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __host__ __device__ uint Device::Rasteriser::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return GetWorldBBox().Contains(viewCtx.mousePos) ? kDrawableObjectPrecisionDrag : kDrawableObjectInvalidSelect;       
    }

    __host__ __device__ vec4 Device::Rasteriser::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        if (!GetWorldBBox().Contains(pWorld)) { return vec4(0.0f); }

#ifdef __CUDA_ARCH__
        const vec2 pObject = ToObjectSpace(pWorld);
        const ivec2 pPixel = ivec2(vec2(m_params.viewport.dims) * (pObject - m_params.viewport.objectBounds.lower) / m_params.viewport.objectBounds.Dimensions());

        if (!m_params.hasValidScene)
        {
            const float hatch = step(0.5f, fract(0.05f * dot(pObject / viewCtx.dPdXY, vec2(1.f))));
            return vec4(kOne * hatch * 0.1f, 1.f);
        }
        else if (pPixel.x >= 0 && pPixel.x < m_params.viewport.dims.x && pPixel.y >= 0 && pPixel.y < m_params.viewport.dims.y)
        {
            return m_objects.frameBuffer->At(pPixel);  
        }      
#else
        return vec4(1.);
#endif
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __host__ AssetHandle<Host::GenericObject> Host::Rasteriser::Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects)
    {
        return AssetAllocator::CreateChildAsset<Host::Rasteriser>(parentAsset, id, genericObjects);
    }

    __host__ Host::Rasteriser::Rasteriser(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects) :
        DrawableObject(initCtx, &m_hostInstance),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::Rasteriser>(*this))
    {                
        DrawableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::DrawableObject>(cu_deviceInstance));
        
        constexpr int kViewportWidth = 1200;
        constexpr int kViewportHeight = 675;
        
        // Create some Cuda objects
        m_hostFrameBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "framebuffer", kViewportWidth, kViewportHeight, nullptr);

        m_deviceObjects.frameBuffer = m_hostFrameBuffer->GetDeviceInstance();

        const vec2 boundHalf = 0.25 * ((kViewportHeight > kViewportWidth) ?
                                      vec2(1.f, float(kViewportHeight) / float(kViewportWidth)) :
                                      vec2(float(kViewportWidth) / float(kViewportHeight), 1.f));

        m_params.viewport.dims = ivec2(kViewportWidth, kViewportHeight);
        m_params.viewport.objectBounds = BBox2f(-boundHalf, boundHalf);
        m_params.frameIdx = 0;
        m_params.wallTime = 0.f;
        m_wallTime.Reset();
    }

    __host__ Host::Rasteriser::~Rasteriser() noexcept
    {
        m_hostFrameBuffer.DestroyAsset();
        m_hostTransforms.DestroyAsset();

        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }    

    __host__ void Host::Rasteriser::OnSynchroniseDrawableObject(const uint syncFlags)
    {
        // Only sync the objects if a SceneContainer has been bound
        if (syncFlags & kSyncObjects)
        { 
            SynchroniseObjects<Device::Rasteriser>(cu_deviceInstance, m_deviceObjects); 
        }
        if (syncFlags & kSyncParams) 
        { 
            SynchroniseObjects<Device::Rasteriser>(cu_deviceInstance, m_params); 
            m_hostInstance.Synchronise(m_params);
        }
    }

    __host__ void Host::Rasteriser::Bind(GenericObjectContainer& objects)
    {
        m_hostSceneContainer = objects.FindFirstOfType<Host::SceneContainer>();
        if (!m_hostSceneContainer)
        {
            Log::Warning("Warning! Path tracer '%s' could not bind to a valid SceneContainer object.", GetAssetID());
            m_params.hasValidScene = false;
        }
        else
        {
            m_deviceObjects.tracables = m_hostSceneContainer->Tracables().GetDeviceInstance();
            
            if (m_hostSceneContainer->Cameras().empty())
            {
                Log::Warning("Warning! Path tracer '%s' found no cameras in the scene.");
                m_hostActiveCamera = nullptr;
            }
            else
            {
                m_hostActiveCamera = m_hostSceneContainer->Cameras().back();
                m_deviceObjects.activeCamera = m_hostActiveCamera->GetDeviceInstance();
            }       

            m_params.hasValidScene = true;
        }

        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ void Host::Rasteriser::Render()
    {        
        if (!m_hostSceneContainer || !m_hostActiveCamera) { return; }

        if (!IsClean()) { Synchronise(kSyncParams); }
        
        //KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        //if (m_params.frameIdx > 10) return;

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostFrameBuffer, blockSize, gridSize);

        // Accumulate the frame
        KernelRender << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);

        // Denoise if necessary
        /*if (m_params.frameIdx % 500 == 0)
        {
            KernelDenoise << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);
        }*/

        IsOk(cudaDeviceSynchronize());

        // If there's no user interaction, signal the viewport to update intermittently to save compute
        constexpr float kViewportUpdateInterval = 1. / 2.f;
        if (m_redrawTimer.Get() > kViewportUpdateInterval)
        {
            SignalDirty(kDirtyViewportRedraw);
            m_redrawTimer.Reset();
        }

        if (m_renderTimer.Get() > 1.)
        {
            Log::Debug("Frame: %i", m_params.frameIdx);
            m_renderTimer.Reset();
        }
    }

    __host__ void Host::Rasteriser::Clear()
    {
        m_hostFrameBuffer->Clear(vec4(0.f));

        m_params.frameIdx = 0;  
        Synchronise(kSyncParams);
    }

    __host__ bool Host::Rasteriser::OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject)
    {
        if (stateID == "kCreateDrawableObjectOpen" || stateID == "kCreateDrawableObjectHover")
        {
            m_isConstructed = true;
            m_isFinalised = true;
            if (stateID == "kCreateDrawableObjectOpen") { Log::Success("Opened path tracer %s", GetAssetID()); }

            return true;
        }
        else if (stateID == "kCreateDrawableObjectAppend")
        {
            m_isFinalised = true;
            return true;
        }

        return false;
    }

    __host__ bool Host::Rasteriser::OnRebuildDrawableObject()
    {
        /*m_scene = m_componentContainer->GenericObjects().FindFirstOfType<Host::SceneContainer>();
        if (!m_scene)
        {
            Log::Warning("Warning: path tracer '%s' expected an initialised scene container but none was found.");
        }*/
        
        return true;
    }

    __host__ uint Host::Rasteriser::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return m_hostInstance.OnMouseClick(viewCtx);
    }

    __host__ BBox2f Host::Rasteriser::ComputeObjectSpaceBoundingBox()
    {
        return m_params.viewport.objectBounds;
    }

    __host__ bool Host::Rasteriser::Serialise(Json::Node& node, const int flags) const
    {
        DrawableObject::Serialise(node, flags);

        Json::Node lookNode = node.AddChildObject("viewport");
        lookNode.AddVector("dims", m_params.viewport.dims);

        return true;
    }

    __host__ bool Host::Rasteriser::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = DrawableObject::Deserialise(node, flags);
        
        Json::Node viewportNode = node.GetChildObject("viewport", flags);
        if (viewportNode)
        {
            isDirty |= viewportNode.GetVector("dims", m_params.viewport.dims, flags);
        }

        if (isDirty)
        {
            SignalDirty({ kDirtyParams });
        }

        return isDirty;
    }

}