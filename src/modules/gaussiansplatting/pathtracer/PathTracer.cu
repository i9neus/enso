#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "PathTracer.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/3d/Ctx.cuh"
#include "core/AssetContainer.cuh"
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

    __device__ void Device::PathTracer::Render()
    {
        CudaAssertDebug(m_objects.accumBuffer);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_objects.accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_objects.accumBuffer->Height()) { return; }        

        RenderCtx renderCtx;
        renderCtx.rng.Initialise(HashOf(m_params.frameIdx, xyScreen.x, xyScreen.y));

        m_objects.accumBuffer->At(xyScreen) = vec4(renderCtx.rng.Rand().xyz, 1.);
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

        m_deviceObjects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        //m_deviceObjects.scene = m_scene->GetDeviceInstance();

        cu_deviceInstance = AssetAllocator::InstantiateOnDevice<Device::PathTracer>(*this);
        
        m_params.viewport.dims = ivec2(width, height);
        m_params.frameIdx = 0;
        Synchronise(kSyncObjects | kSyncParams);
    }

    Host::PathTracer::~PathTracer() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
        m_hostAccumBuffer.DestroyAsset();
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