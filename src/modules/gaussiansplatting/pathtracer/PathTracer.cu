#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "PathTracer.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/3d/Ctx.cuh"
#include "core/AssetContainer.cuh"
#include "core/3d/Transform.cuh"
#include "core/Vector.cuh"
#include "Geometry.cuh"
#include "core/3d/Cameras.cuh"
#include "Scene.cuh"
#include "Integrator.cuh"
#include "../ComponentContainer.cuh"
#include "../scene/SceneContainer.cuh"

#include "io/json/JsonUtils.h"
//#include "core/AccumulationBuffer.cuh"

namespace Enso
{        
    __host__ __device__ PathTracerParams::PathTracerParams()
    {
        viewport.dims = ivec2(0);  
        frameIdx = 0;
    }

    __host__ __device__ void PathTracerParams::Validate() const
    {
        CudaAssert(viewport.dims.x != 0 && viewport.dims.y != 0);
    }   

    __host__ __device__ void Device::PathTracer::Synchronise(const PathTracerParams& params) 
    {
        m_params = params;   
        m_nlm.Initialise(10, 2, 2.f, 2.f);
     }
    
    __device__ void Device::PathTracer::Synchronise(const PathTracerObjects& objects) 
    { 
        objects.Validate(); 
        m_objects = objects; 
        m_nlm.Initialise(m_objects.meanAccumBuffer, m_objects.varAccumBuffer);
    }

    __device__ void Device::PathTracer::Render()
    {
        CudaAssertDebug(m_objects.transforms->Size() == 9);

        const ivec2 xyViewport = kKernelPos<ivec2>();
        if (xyViewport.x < 0 || xyViewport.x >= m_params.viewport.dims.x || xyViewport.y < 0 || xyViewport.y >= m_params.viewport.dims.y) { return; }        

        // Get pointers to the object transforms
        const auto& transforms = *m_objects.transforms;
        const auto& emitterTrans = transforms.Back();

        // Create a render context
        RenderCtx renderCtx;
        renderCtx.rng.Initialise(HashOf(m_params.frameIdx, xyViewport.x, xyViewport.y));
        renderCtx.qrng.Initialise(0x7a67bbfc, HashOf(xyViewport.x, xyViewport.y) + m_params.frameIdx);
        renderCtx.viewport.dims = m_params.viewport.dims;
        renderCtx.viewport.xy = xyViewport;
        renderCtx.frameIdx = m_params.frameIdx;

        // Transform into normalised sceen space
        const vec2 uvView = ScreenToNormalisedScreen(vec2(xyViewport) + renderCtx.Rand(0).xy, vec2(m_params.viewport.dims));
        Ray directRay, indirectRay;

        // Create the camera ray
        float cameraPhi = -kPi;
        //cameraPhi += kTwoPi * m_params.wallTime * 0.01f;
        vec3 cameraPos = vec3(cos(cameraPhi), 0.5, sin(cameraPhi)) * 2.;
        indirectRay = Cameras::CreatePinholeRay(uvView, cameraPos, vec3(0., -0., -0.), 50.);

        int genFlags = kGeneratedIndirect;
        HitCtx hit;
        vec3 L = kZero;
        //int renderMode = (xyViewport.x < m_params.viewport.dims.x * 0.5f) ? kModePathTraced : kModeNEE;
        const int renderMode = kModeNEE;

        constexpr int kMaxPathDepth = 5;
        constexpr int kMaxIterations = 10;
        for (int rayIdx = 0; rayIdx < kMaxIterations && genFlags != kGenerateNothing; ++rayIdx)
        {
            Ray ray;
            if ((genFlags & kGeneratedDirect) != 0) ray = directRay; else ray = indirectRay;

            if (ray.depth >= kMaxPathDepth) { continue; }

            if (Trace(ray, hit, transforms) == kMatInvalid && !ray.IsDirectSample())
            {
                //if(depth > 0)
                {
                    L += kOne * ray.weight * 0.2;
                    //L += kOne * 0.5 * luminance(texture(iChannel1, ray.od.d).xyz);
                }
                break;
            }

            //EvaluateMaterial(ray, hit);

            //L = hit.n * 0.5 + 0.5;
            //break;

            if ((genFlags & kGeneratedDirect) != 0)
            {
                ShadeDirectSample(ray, hit, L);
                genFlags &= ~kGeneratedDirect;
            }
            else if ((genFlags & kGeneratedIndirect) != 0)
            {
                genFlags = Shade(ray, indirectRay, directRay, hit, renderCtx, emitterTrans, renderMode, L);
            }
        }

        auto& meanL = m_objects.meanAccumBuffer->At(xyViewport);
        auto& varL = m_objects.varAccumBuffer->At(xyViewport);

        // Unstable running variance
        //varL += vec4(sqr(L), 1.);
        
        // Welford's online algorithm
        varL += vec4((L - meanL.xyz / fmaxf(1.f, meanL.w)) * 
                     (L - (L + meanL.xyz) / (1.f + meanL.w)), 1.0f);

        meanL += vec4(L, 1.);

    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::PathTracer::Denoise()
    {        
        const ivec2 xyViewport = kKernelPos<ivec2>();
        if (xyViewport.x < m_params.viewport.dims.x && xyViewport.y < m_params.viewport.dims.y)
        {
            m_objects.denoisedBuffer->At(xyViewport) = m_nlm.FilterPixel(xyViewport);
        }
    }
    DEFINE_KERNEL_PASSTHROUGH(Denoise);

    __device__ void Device::PathTracer::Composite(Device::ImageRGBA* deviceOutputImage)
    {
        CudaAssertDebug(deviceOutputImage);

        // TODO: Make alpha compositing a generic operation inside the Image class.
        const ivec2 xyAccum = kKernelPos<ivec2>();
        const ivec2 xyViewport = xyAccum + deviceOutputImage->Dimensions() / 2 - m_objects.meanAccumBuffer->Dimensions() / 2;
        BBox2i border(0, 0, m_params.viewport.dims.x, m_params.viewport.dims.y);
        /*if(border.PointOnPerimiter(xyAccum, 2))
        {
            deviceOutputImage->At(xyViewport) = vec4(1.0f);
        }*/
        if (xyAccum.x < m_params.viewport.dims.x && xyAccum.y < m_params.viewport.dims.y)
        {
            if (xyAccum.x < m_params.viewport.dims.x / 2)
            {
                //const vec4& varL = m_objects.varAccumBuffer->At(xyAccum);
                const vec4& meanL = m_objects.meanAccumBuffer->At(xyAccum);
                deviceOutputImage->At(xyViewport) = vec4(meanL.xyz / fmaxf(1.f, meanL.w), 1.0f);

                //deviceOutputImage->At(xyViewport) = vec4(varL.xyz / fmaxf(1.f, varL.w) - sqr(meanL.xyz / fmaxf(1.f, meanL.w)), 1.f);
                //deviceOutputImage->At(xyViewport) = vec4(varL.xyz / sqr(fmaxf(1.f, varL.w)), 1.f);
            }
            else
            {
                const vec3& denoisedL = m_objects.denoisedBuffer->At(xyAccum);
                deviceOutputImage->At(xyViewport) = vec4(denoisedL, 1.f);
            }
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    __host__ __device__ uint Device::PathTracer::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return GetWorldBBox().Contains(viewCtx.mousePos) ? kDrawableObjectPrecisionDrag : kDrawableObjectInvalidSelect;       
    }

    __host__ __device__ vec4 Device::PathTracer::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        if (!GetWorldBBox().Contains(pWorld)) { return vec4(0.0f); }

#ifdef __CUDA_ARCH__
        const vec2 pObject = ToObjectSpace(pWorld);

        const ivec2 pPixel = ivec2(vec2(m_params.viewport.dims) * (pObject - m_params.viewport.objectBounds.lower) / m_params.viewport.objectBounds.Dimensions());

        if (pPixel.x >= 0 && pPixel.x < m_params.viewport.dims.x && pPixel.y >= 0 && pPixel.y < m_params.viewport.dims.y)
        {
            if (pPixel.x < m_params.viewport.dims.x / 2)
            {
                //const vec4& varL = m_objects.varAccumBuffer->At(xyAccum);
                const vec4& meanL = m_objects.meanAccumBuffer->At(pPixel);
                return vec4(meanL.xyz / fmaxf(1.f, meanL.w), 1.0f);

                //deviceOutputImage->At(xyViewport) = vec4(varL.xyz / fmaxf(1.f, varL.w) - sqr(meanL.xyz / fmaxf(1.f, meanL.w)), 1.f);
                //deviceOutputImage->At(xyViewport) = vec4(varL.xyz / sqr(fmaxf(1.f, varL.w)), 1.f);
            }
            else
            {
                const vec3& denoisedL = m_objects.denoisedBuffer->At(pPixel);
                return vec4(denoisedL, 1.f);
            }
        }      
#else
        return vec4(1.);
#endif
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __host__ AssetHandle<Host::GenericObject> Host::PathTracer::Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::ComponentContainer>& scene)
    {
        return AssetAllocator::CreateChildAsset<Host::PathTracer>(parentAsset, id, scene);
    }

    Host::PathTracer::PathTracer(const Asset::InitCtx& initCtx, const AssetHandle<const Host::ComponentContainer>& scene):
        DrawableObject(initCtx, &m_hostInstance, scene),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::PathTracer>(*this)),
        m_scene(scene)
    {                
        DrawableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::DrawableObject>(cu_deviceInstance));
        
        constexpr int kViewportWidth = 1200;
        constexpr int kViewportHeight = 675;
        
        // Create some Cuda objects
        m_hostMeanAccumBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "meanAccumBufferMean", kViewportWidth, kViewportHeight, nullptr);
        m_hostVarAccumBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "meanAccumBufferVar", kViewportWidth, kViewportHeight, nullptr);
        m_hostDenoisedBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGB>(*this, "denoisedBuffer", kViewportWidth, kViewportHeight, nullptr);
        m_hostTransforms = AssetAllocator::CreateChildAsset<Host::Vector<BidirectionalTransform>>(*this, "transforms", kVectorHostAlloc);
        m_hostSceneContainer = AssetAllocator::CreateChildAsset<Host::SceneContainer>(*this, "scene");

        m_deviceObjects.meanAccumBuffer = m_hostMeanAccumBuffer->GetDeviceInstance();
        m_deviceObjects.varAccumBuffer = m_hostVarAccumBuffer->GetDeviceInstance();
        m_deviceObjects.denoisedBuffer = m_hostDenoisedBuffer->GetDeviceInstance();
        m_deviceObjects.transforms = m_hostTransforms->GetDeviceInstance();
        //m_deviceObjects.scene = m_scene->GetDeviceInstance();

        const vec2 boundHalf = 0.25 * ((kViewportHeight > kViewportWidth) ?
                                      vec2(1.f, float(kViewportHeight) / float(kViewportWidth)) :
                                      vec2(float(kViewportWidth) / float(kViewportHeight), 1.f));

        m_params.viewport.dims = ivec2(kViewportWidth, kViewportHeight);
        m_params.viewport.objectBounds = BBox2f(-boundHalf, boundHalf);
        m_params.frameIdx = 0;
        m_params.wallTime = 0.f;
        m_wallTime.Reset();

        CreateScene();
    }

    Host::PathTracer::~PathTracer() noexcept
    {
        m_hostMeanAccumBuffer.DestroyAsset();
        m_hostVarAccumBuffer.DestroyAsset();
        m_hostDenoisedBuffer.DestroyAsset();
        m_hostTransforms.DestroyAsset();

        m_hostSceneContainer->DestroyManagedObjects();
        m_hostSceneContainer.DestroyAsset();

        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::PathTracer::CreateScene()
    {
        m_hostTransforms->Clear();

#define kNumSpheres 7        
        for (int sphereIdx = 0; sphereIdx < kNumSpheres; ++sphereIdx)
        {
            float phi = kTwoPi * (0.75f + float(sphereIdx) / float(kNumSpheres));
            m_hostTransforms->PushBack(BidirectionalTransform(vec3(cos(phi), 0.f, sin(phi)) * 0.7f, kZero, 0.2f));
        }

        m_hostTransforms->PushBack(BidirectionalTransform(vec3(0.f, -0.2f, 0.f), vec3(-kHalfPi, 0.f, 0.f), 2.f));   // Ground plane
        m_hostTransforms->PushBack(BidirectionalTransform(kEmitterPos, kEmitterRot, kEmitterSca));                  // Emitter plane

        m_hostTransforms->Synchronise(kVectorSyncUpload);

        Synchronise(kSyncObjects | kSyncParams);
    }

    __host__ void Host::PathTracer::OnSynchroniseDrawableObject(const uint syncFlags)
    {
        if (syncFlags & kSyncObjects) { SynchroniseObjects<Device::PathTracer>(cu_deviceInstance, m_deviceObjects); }
        if (syncFlags & kSyncParams) 
        { 
            SynchroniseObjects<Device::PathTracer>(cu_deviceInstance, m_params); 
            m_hostInstance.Synchronise(m_params);
        }
    }

    __host__ void Host::PathTracer::Render()
    {
        //KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        //if (m_params.frameIdx > 10) return;

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostMeanAccumBuffer, blockSize, gridSize);

        // Accumulate the frame
        KernelRender << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);

        // Denoise if necessary
        if (m_params.frameIdx % 500 == 0)
        {
            KernelDenoise << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);
        }

        IsOk(cudaDeviceSynchronize());

        if (m_renderTimer.Get() > 1.)
        {
            m_renderTimer.Reset();
            Log::Debug("Frame: %i", m_params.frameIdx);
        }
    }

    __host__ void Host::PathTracer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostMeanAccumBuffer, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }

    __host__ bool Host::PathTracer::Prepare()
    {
        m_params.frameIdx++;
        m_params.wallTime = m_wallTime.Get();

        // Upload to the device
        Synchronise(kSyncParams);
        return true;
    }

    __host__ void Host::PathTracer::Clear()
    {
        m_hostMeanAccumBuffer->Clear(vec4(0.f));

        m_params.frameIdx = 0;  
        Synchronise(kSyncParams);
    }

    __host__ bool Host::PathTracer::OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject)
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

    __host__ bool Host::PathTracer::OnRebuildDrawableObject()
    {
        return true;
    }

    __host__ uint Host::PathTracer::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return m_hostInstance.OnMouseClick(viewCtx);
    }

    __host__ BBox2f Host::PathTracer::ComputeObjectSpaceBoundingBox()
    {
        Log::Debug("%s", m_params.viewport.objectBounds.Format());
        return m_params.viewport.objectBounds;
    }

    __host__ bool Host::PathTracer::Serialise(Json::Node& node, const int flags) const
    {
        DrawableObject::Serialise(node, flags);

        Json::Node lookNode = node.AddChildObject("viewport");
        lookNode.AddVector("dims", m_params.viewport.dims);

        return true;
    }

    __host__ bool Host::PathTracer::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = DrawableObject::Deserialise(node, flags);
        
        Json::Node viewportNode = node.GetChildObject("viewport", flags);
        if (viewportNode)
        {
            isDirty |= viewportNode.GetVector("dims", m_params.viewport.dims, flags);
        }

        if (isDirty)
        {
            SignalDirty({ kDirtyParams, kDirtyIntegrators });
        }

        return isDirty;
    }

}