#include "CudaGI2DPathTracer.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"

#include "../RenderCtx.cuh"

using namespace Cuda;

namespace GI2D
{        
    __host__ __device__ PathTracerParams::PathTracerParams()
    {
        m_accum.width = 0;
        m_accum.height = 0;
        m_accum.downsample = 1;
        m_frameIdx = 0;
    }

    __device__ Device::PathTracer::PathTracer() { }

    __device__ void Device::PathTracer::Render()
    {
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_accumBuffer->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_viewCtx.transform.matrix * vec2(xyScreen * m_accum.downsample);

        vec4& accum = m_accumBuffer->At(xyScreen);
        vec3 L(0.f);

        if (!m_viewCtx.sceneBounds.Contains(xyView)) { accum = vec4(0.0f, 0.0f, 0.0f, 1.0f); return; }

        assert(m_bih);
        const auto& bih = *m_bih;
        const VectorInterface<TracableInterface*>& tracables = *m_tracables;
        RNG rng;       
        rng.Initialise(HashOf(uint(kKernelY * kKernelWidth + kKernelX), uint(accum.w)));
        //rng.Initialise(HashOf(uint(accum.w)));

        for (int depth = 0; depth < 1; ++depth)
        {
            float theta = rng.Rand<0>() * kTwoPi;
            //const float theta = kTwoPi * (accum.w + rng.Rand<0>()) / 100.0f;
            Ray2D ray(xyView, vec2(cos(theta), sin(theta)));
            HitCtx2D hit;
            int hitIdx = 0;

            auto onIntersect = [&](const uint* primRange, RayRange2D& range) -> float
            {
                for (uint idx = primRange[0]; idx < primRange[1]; ++idx)
                {
                    if (tracables[idx]->IntersectRay(ray, hit))
                    {
                        if (hit.tFar < range.tFar)
                        {
                            range.tFar = hit.tFar;
                            hitIdx = idx;
                        }
                    }
                }
            };
            bih.TestRay(ray, kFltMax, onIntersect);

            if (hit.tFar != kFltMax)
            {
                L += tracables[hitIdx]->GetColour();
            }
        }

        accum.xyz += L;
        accum.w += 1.0f;
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::PathTracer::Composite(Cuda::Device::ImageRGBA* deviceOutputImage)
    {
        assert(deviceOutputImage);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= deviceOutputImage->Width() || xyScreen.y < 0 || xyScreen.y >= deviceOutputImage->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_viewCtx.transform.matrix * vec2(xyScreen);

        if (!m_viewCtx.sceneBounds.Contains(xyView)) 
        { 
            deviceOutputImage->At(xyScreen) = vec4(0.1f, 0.1f, 0.1f, 1.0f);
            return; 
        }

        const vec2 uv = vec2(xyScreen) * vec2(m_accumBuffer->Dimensions()) / vec2(deviceOutputImage->Dimensions());
        vec4 L = m_accumBuffer->Lerp(uv);

        deviceOutputImage->At(xyScreen) = vec4(L.xyz / fmaxf(L.w, 1.0f), 1.0f);
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    Host::PathTracer::PathTracer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables,
                                 const uint width, const uint height, const uint downsample, cudaStream_t renderStream) :
        UILayer(id, bih, tracables)
    {
        // Create some Cuda objects
        m_hostAccumBuffer = CreateChildAsset<Cuda::Host::ImageRGBW>("id_2dgiAccumBuffer", width / downsample, height / downsample, renderStream);

        m_deviceObjects.m_bih = m_hostBIH->GetDeviceInstance();
        m_deviceObjects.m_tracables = m_hostTracables->GetDeviceInterface();
        m_deviceObjects.m_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();

        m_accum.width = width;
        m_accum.height = height;
        m_accum.downsample = downsample;

        cu_deviceData = InstantiateOnDevice<Device::PathTracer>();

        Synchronise(kSyncObjects);
    }

    Host::PathTracer::~PathTracer()
    {
        OnDestroyAsset();
    }


    __host__ void Host::PathTracer::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {
        UILayer::Rebuild(dirtyFlags, viewCtx, selectionCtx);

        Synchronise(kSyncParams);
    }

    __host__ void Host::PathTracer::Synchronise(const int syncType)
    {
        UILayer::Synchronise(cu_deviceData, syncType);

        if (syncType & kSyncObjects) { SynchroniseObjects2<PathTracerObjects>(cu_deviceData, m_deviceObjects); }
        if (syncType & kSyncParams)  { SynchroniseObjects2<PathTracerParams>(cu_deviceData, *this); }
    }

    __host__ void Host::PathTracer::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::PathTracer::Render()
    {
        if (m_dirtyFlags)
        {
            m_hostAccumBuffer->Clear(vec4(0.0f));
            m_dirtyFlags = 0;
        }

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelRender << < gridSize, blockSize, 0 >> > (cu_deviceData);
        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::PathTracer::Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(hostOutputImage, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0 >> > (cu_deviceData, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }
}