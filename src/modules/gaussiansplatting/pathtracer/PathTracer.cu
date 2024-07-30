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
//#include "../scene/SceneContainer.cuh"

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

    __host__ __device__ void PathTracerObjects::Validate() const
    {
        CudaAssert(transforms);
        CudaAssert(accumBuffer);
    }

    __device__ void Device::PathTracer::Render()
    {
        CudaAssertDebug(m_objects.accumBuffer);
        CudaAssertDebug(m_objects.transforms->Size() == 9);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_objects.accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_objects.accumBuffer->Height()) { return; }        

        const auto& transforms = *m_objects.transforms;

        RenderCtx renderCtx;
        renderCtx.rng.Initialise(HashOf(m_params.frameIdx, xyScreen.x, xyScreen.y));

        const vec2 uvView = ScreenToNormalisedScreen(vec2(xyScreen) + renderCtx.Rand().xy, vec2(m_params.viewport.dims));

        float cameraPhi = -kPi;
        //cameraPhi += kTwoPi * mix(-1., 1., iMouse.x / iResolution.x);
        cameraPhi += kTwoPi * m_params.wallTime * 0.01f;
        //float cameraTheta = kHalfPi * iMouse.y / iResolution.y;
        vec3 cameraPos = vec3(cos(cameraPhi), 0.5, sin(cameraPhi)) * 2.;//mix(0.5, 3., iMouse.y / iResolution.y);
        Ray ray = CreatePinholeCameraRay(uvView, cameraPos, vec3(0., -0., -0.), 50.f);
        HitCtx hit;
        vec3 L = kZero;
        bool referenceMode = true;//(xyScreen.x < iResolution.x * 0.5);

        #define kMaxPathDepth 4
        for (int depth = 0; depth < kMaxPathDepth; ++depth)
        {
            if (Trace(ray, hit, transforms) == kMatInvalid && !ray.IsDirectSample())
            {
                //if (depth > 0)
                //    L += kOne * luminance(texture(iChannel1, ray.od.d).xyz) * ray.weight * 0.5;
                break;
            }


            L = hit.n *0.5f + 0.5f;
            break;

            /*if (!Shade(ray, hit, renderCtx, emitterTrans, depth, referenceMode, L))
            {
                break;
            }*/
        }

        m_objects.accumBuffer->At(xyScreen) = vec4(L, 1.);
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::PathTracer::Composite(Device::ImageRGBA* deviceOutputImage)
    {
        CudaAssertDebug(deviceOutputImage);

        // TODO: Make alpha compositing a generic operation inside the Image class.
        const ivec2 xyAccum = kKernelPos<ivec2>();
        const ivec2 xyScreen = xyAccum + deviceOutputImage->Dimensions() / 2 - m_objects.accumBuffer->Dimensions() / 2;
        BBox2i border(0, 0, m_params.viewport.dims.x, m_params.viewport.dims.y);
        if(border.PointOnPerimiter(xyAccum, 2))
        {
            deviceOutputImage->At(xyScreen) = vec4(1.0f);
        }
        else if (xyAccum.x < m_objects.accumBuffer->Width() && xyAccum.y < m_objects.accumBuffer->Height())
        {
            const vec4& L = m_objects.accumBuffer->At(xyAccum);
            deviceOutputImage->At(xyScreen) = vec4(L.xyz / fmaxf(1.f, L.w), 1.f);
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    Host::PathTracer::PathTracer(const Asset::InitCtx& initCtx, /*const AssetHandle<const Host::SceneContainer>& scene, */const uint width, const uint height, cudaStream_t renderStream):
        GenericObject(initCtx)
        //m_scene(scene)
    {                
        // Create some Cuda objects
        m_hostAccumBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "accumBuffer", width, height, renderStream);
        m_hostTransforms = AssetAllocator::CreateChildAsset<Host::Vector<BidirectionalTransform>>(*this, "transforms", kVectorHostAlloc);

        m_deviceObjects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.transforms = m_hostTransforms->GetDeviceInstance();
        //m_deviceObjects.scene = m_scene->GetDeviceInstance();

        cu_deviceInstance = AssetAllocator::InstantiateOnDevice<Device::PathTracer>(*this);
        
        m_params.viewport.dims = ivec2(width, height);
        m_params.frameIdx = 0;
        m_params.wallTime = 0.f;
        m_wallTime.Reset();

        CreateScene();
    }    

    Host::PathTracer::~PathTracer() noexcept
    {
        m_hostAccumBuffer.DestroyAsset();
        m_hostTransforms.DestroyAsset();
        
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
        m_hostAccumBuffer.DestroyAsset();
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

        m_hostTransforms->PushBack(BidirectionalTransform(vec3(0.f, -0.2f, 0.f), vec3(kHalfPi, 0.f, 0.f), 2.f));   // Ground plane
        m_hostTransforms->PushBack(BidirectionalTransform(kEmitterPos, kEmitterRot, kEmitterSca));                  // Emitter plane

        m_hostTransforms->Synchronise(kVectorSyncUpload);

        Synchronise(kSyncObjects | kSyncParams);
    }

    __host__ void Host::PathTracer::Synchronise(const uint syncFlags)
    {
        if (syncFlags & kSyncObjects) { SynchroniseObjects<Device::PathTracer>(cu_deviceInstance, m_deviceObjects); }
        if (syncFlags & kSyncParams) { SynchroniseObjects<Device::PathTracer>(cu_deviceInstance, m_params); }
    }

    __host__ void Host::PathTracer::Render()
    {
        //KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelRender << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);
        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::PathTracer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

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
        m_hostAccumBuffer->Clear(vec4(0.f));

        m_params.frameIdx = 0;  
        Synchronise(kSyncParams);
    }
}