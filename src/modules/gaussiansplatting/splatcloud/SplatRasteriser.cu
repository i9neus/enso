#include "SplatRasteriser.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/3d/Ctx.cuh"
#include "core/assets/AssetContainer.cuh"
#include "core/3d/Transform.cuh"
#include "core/containers/Vector.cuh"
#include "core/3d/Cameras.cuh"
#include "core/assets/GenericObjectContainer.cuh"
#include "core/math/samplers/MersenneTwister.cuh"
#include "core/math/mat/MatMul.cuh"

#include "../scene/SceneContainer.cuh"
#include "../scene/cameras/Camera.cuh"
#include "../scene/materials/Material.cuh"
#include "../scene/lights/Light.cuh"
#include "../scene/textures/Texture2D.cuh"
#include "../scene/tracables/Tracable.cuh"
#include "../scene/pointclouds/GaussianPointCloud.cuh"

#include "io/json/JsonUtils.h"
//#include "core/AccumulationBuffer.cuh"

namespace Enso
{        
    __host__ __device__ void Device::SplatRasteriser::Synchronise(const SplatRasteriserParams& params) 
    {
        m_params = params;   
     }
    
    __device__ void Device::SplatRasteriser::Synchronise(const SplatRasteriserObjects& objects) 
    { 
        m_objects = objects;
    }

    __device__ void Device::SplatRasteriser::Render()
    {
        const ivec2 xyViewport = kKernelPos<ivec2>();
        if (xyViewport.x < 0 || xyViewport.x >= m_params.viewport.dims.x || xyViewport.y < 0 || xyViewport.y >= m_params.viewport.dims.y) { return; }

        // Load some data into shared memory and sync the block
        __shared__ mat3 W;
        __shared__ vec3 camPos;
        __shared__ float camFov;
        if (kThreadIdx == 0)
        {
            const auto& params = m_objects.activeCamera->GetCameraParams();
            W = params.inv;
            camPos = params.cameraPos;
            camFov = params.cameraFov;
        }
        __syncthreads();

        const vec2 uvView = ScreenToNormalisedScreen(vec2(xyViewport), vec2(m_params.viewport.dims));
        vec3 L = kZero;       

        for (int idx = 0; idx < m_objects.pointCloud->size(); ++idx)
        {
            const auto& splat = (*m_objects.pointCloud)[idx];

            // Project the position of the splat into camera space
            const vec3 pCam = W * (splat.p - camPos);
            const vec3 pView = pCam / (pCam.z * -tanf(toRad(camFov)));

            // Create rotation and transpose product of scale matrices
            const mat3 R = splat.rot.RotationMatrix();
            const mat3 ST(vec3(splat.sca.x * splat.sca.x, 0.0f, 0.0f), 
                          vec3(0., splat.sca.y* splat.sca.y, 0.0), 
                          vec3(0., 0.0, splat.sca.z* splat.sca.z));

            // Jacobian of projective approximation (Zwicker et al)
            const float lenPCam = length(pCam);
            const mat3 J = mat3(vec3(1. / pCam.z, 0.0, pCam.x / lenPCam),
                           vec3(0., 1. / pCam.z, pCam.y / lenPCam),
                           vec3(-pCam.x / (pCam.z * pCam.z), -pCam.y / (pCam.z * pCam.z), pCam.z / lenPCam));

            // Build covariance matrix
            const mat3 cov = R * ST * transpose(R);

            // Project covariance matrix
            const mat3 sigma3 = J * W * cov * transpose(W) * transpose(J);
            const mat2 sigma2 = mat2(sigma3[0].xy, sigma3[1].xy);

            // Gaussian PDF
            const vec2 mu = uvView - pView.xy;
            const float G = expf(-0.5 * dot(mu * inverse(sigma2), mu));

            // Splat 
            L = mix(L, splat.rgba.xyz, G * splat.rgba.w);
        }

        m_objects.frameBuffer->At(xyViewport) = vec4(L, 1.0f);
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __host__ __device__ vec4 Device::SplatRasteriser::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        if (!GetWorldBBox().Contains(pWorld)) { return vec4(0.0f); }

#ifdef __CUDA_ARCH__
        const vec2 pObject = ToObjectSpace(pWorld);
        const ivec2 pPixel = ivec2(vec2(m_params.viewport.dims) * (pObject - m_params.viewport.objectBounds.lower) / m_params.viewport.objectBounds.Dimensions());

        if (!m_params.hasValidSplatCloud)
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

    __host__ AssetHandle<Host::GenericObject> Host::SplatRasteriser::Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects)
    {
        return AssetAllocator::CreateChildAsset<Host::SplatRasteriser>(parentAsset, id, genericObjects);
    }

    __host__ Host::SplatRasteriser::SplatRasteriser(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects) :
        DrawableObject(initCtx, &m_hostInstance),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::SplatRasteriser>(*this))
    {                        
        DrawableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::DrawableObject>(cu_deviceInstance));
        RenderableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::RenderableObject>(cu_deviceInstance));
        
        constexpr int kViewportWidth = 1200;
        constexpr int kViewportHeight = 675;
        
        // Create some Cuda objects
        m_hostFrameBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "framebuffer", kViewportWidth, kViewportHeight, nullptr);

        m_objects.frameBuffer = m_hostFrameBuffer->GetDeviceInstance();

        const vec2 boundHalf = 0.25 * ((kViewportHeight > kViewportWidth) ?
                                      vec2(1.f, float(kViewportHeight) / float(kViewportWidth)) :
                                      vec2(float(kViewportWidth) / float(kViewportHeight), 1.f));

        m_params.viewport.dims = ivec2(kViewportWidth, kViewportHeight);
        m_params.viewport.objectBounds = BBox2f(-boundHalf, boundHalf);
        m_wallTime.Reset();

        Cascade({ kDirtySceneObjectChanged });
    }

    __host__ Host::SplatRasteriser::~SplatRasteriser() noexcept
    {
        m_hostFrameBuffer.DestroyAsset();
        m_hostTransforms.DestroyAsset();

        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }    

    __host__ void Host::SplatRasteriser::OnSynchroniseDrawableObject(const uint syncFlags)
    {
        // Only sync the objects if a SceneContainer has been bound
        if (syncFlags & kSyncObjects)
        { 
            SynchroniseObjects<Device::SplatRasteriser>(cu_deviceInstance, m_objects); 
        }
        if (syncFlags & kSyncParams) 
        { 
            SynchroniseObjects<Device::SplatRasteriser>(cu_deviceInstance, m_params); 
            m_hostInstance.Synchronise(m_params);
        }
    }

    __host__ void Host::SplatRasteriser::RebuildSplatCloud()
    {
        if (!m_hostSceneContainer || !m_gaussianPointCloud) { return; }

        Log::Debug("Building splat cloud...");

        auto& tracables = m_hostSceneContainer->Tracables();
        constexpr int kTotalSplats = 1000;
        float totalSurfaceArea = 0.f;
        int numGeneratedSplats = 0;

        std::vector<float> tracableAreas(tracables.size());
        MersenneTwister rng(987659672);

        Log::Debug("Geometry surface area:");
        for (int idx = 0; idx < tracables.size(); ++idx)
        {
            tracableAreas[idx] = tracables[idx]->CalculateSurfaceArea();
            totalSurfaceArea += tracableAreas[idx];
            Log::Debug("  - %s: %f", tracables[idx]->GetAssetID(), tracableAreas[idx]);
        }

        for(int idx = 0; idx < m_hostSceneContainer->Tracables().size(); ++idx)        
        {
            const int numSplats = std::max(1, int(std::ceil(kTotalSplats * tracableAreas[idx] / totalSurfaceArea)));
            auto splatList = tracables[idx]->GenerateGaussianPointCloud(numSplats, rng);
            m_gaussianPointCloud->AppendSplats(splatList);
            numGeneratedSplats += splatList.size();
        }

        Log::Debug("Created cloud containing %i splats", numGeneratedSplats);

        m_gaussianPointCloud->Finalise();
    }

    __host__ void Host::SplatRasteriser::Bind(GenericObjectContainer& objects)
    {
        // REMOVE THIS!!!
        if (m_gaussianPointCloud) return;
         
        m_hostSceneContainer = objects.FindFirstOfType<Host::SceneContainer>();
        if (!m_hostSceneContainer)
        {
            Log::Warning("Warning! Splat rasteriser '%s' could not bind to a valid SceneContainer object.", GetAssetID());
        }
        else
        {           
            if (m_hostSceneContainer->Cameras().empty())
            {
                Log::Warning("Warning! Splat rasteriser '%s' found no cameras in the scene.");
                m_hostActiveCamera = nullptr;
            }
            else
            {
                m_hostActiveCamera = m_hostSceneContainer->Cameras().back();
                m_objects.activeCamera = m_hostActiveCamera->GetDeviceInstance();
            }       
        }

        m_gaussianPointCloud = objects.FindFirstOfType<Host::GaussianPointCloud>();
        if (m_gaussianPointCloud)
        {
            m_objects.pointCloud = m_gaussianPointCloud->GetSplatList().GetDeviceInstance();
            m_params.hasValidSplatCloud = true;

            RebuildSplatCloud();
        }
        else
        {
            Log::Warning("Warning! Splat rasteriser '%s' could not bind to a valid GaussianPointCloud object.", GetAssetID());
            m_objects.pointCloud = nullptr;
            m_params.hasValidSplatCloud = false;
        } 

        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ void Host::SplatRasteriser::Render()
    {        
        if (!m_hostSceneContainer || !m_hostActiveCamera) { return; }

                
        //KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        //if (m_params.frameIdx > 10) return;

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostFrameBuffer, blockSize, gridSize);

        if(RenderableObject::m_params.frameIdx <= 1 || IsDirty(kDirtySceneObjectChanged))
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
            Log::Debug("Frame: %i", RenderableObject::m_params.frameIdx);
            m_renderTimer.Reset();
        }
    }

    __host__ void Host::SplatRasteriser::Clear()
    {
        m_hostFrameBuffer->Clear(vec4(0.f));

        RenderableObject::m_params.frameIdx = 0;
        Synchronise(kSyncParams);
    }

    __host__ bool Host::SplatRasteriser::OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject)
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

    __host__ bool Host::SplatRasteriser::OnRebuildDrawableObject()
    {
        /*m_scene = m_componentContainer->GenericObjects().FindFirstOfType<Host::SceneContainer>();
        if (!m_scene)
        {
            Log::Warning("Warning: path tracer '%s' expected an initialised scene container but none was found.");
        }*/
        
        return true;
    }

    __host__ bool Host::SplatRasteriser::IsClickablePoint(const UIViewCtx& viewCtx) const
    {
        return GetWorldSpaceBoundingBox().Contains(viewCtx.mousePos);
    }   

    __host__ BBox2f Host::SplatRasteriser::ComputeObjectSpaceBoundingBox()
    {
        return m_params.viewport.objectBounds;
    }

    __host__ bool Host::SplatRasteriser::Serialise(Json::Node& node, const int flags) const
    {
        DrawableObject::Serialise(node, flags);

        Json::Node lookNode = node.AddChildObject("viewport");
        lookNode.AddVector("dims", m_params.viewport.dims);

        return true;
    }

    __host__ bool Host::SplatRasteriser::Deserialise(const Json::Node& node, const int flags)
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