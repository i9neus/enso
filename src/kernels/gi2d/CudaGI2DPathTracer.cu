#include "CudaGI2DPathTracer.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"

#include "Ray2D.cuh"
#include "RenderCtx.cuh"

using namespace Cuda;

namespace GI2D
{        
    __host__ __device__ PathTracerParams::PathTracerParams()
    {
        sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));
        isDirty = false;

        accum.width = 0;
        accum.height = 0;
        accum.downsample = 1;
        frameIdx = 0;

        sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));
    }

    __device__ Device::PathTracer::PathTracer(const PathTracerParams& params, const Objects& objects) :
        m_params(params),
        m_objects(objects)
    {
      
    }

    __device__ void Device::PathTracer::Synchronise(const PathTracerParams& params)
    {
        m_params = params;
    }

    __device__ void Device::PathTracer::Synchronise(const Objects& objects)
    {
        m_objects = objects;
    }

    __device__ void Device::PathTracer::Render()
    {
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_objects.accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_objects.accumBuffer->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_params.view.matrix * vec2(xyScreen * m_params.accum.downsample);

        vec4& accum = m_objects.accumBuffer->At(xyScreen);
        vec3 L(0.f);

        if (!m_params.sceneBounds.Contains(xyView)) { accum = vec4(0.0f, 0.0f, 0.0f, 1.0f); return; }

        assert(m_objects.bih);
        const auto& bih = *m_objects.bih;
        const Cuda::Device::Vector<LineSegment>& segments = *m_objects.lineSegments;
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

            auto onIntersect = [&](const uint& startIdx, const uint& endIdx, float& tNearest) -> float
            {
                for (uint idx = startIdx; idx < endIdx; ++idx)
                {
                    if (segments[idx].TestRay(ray, hit))
                    {
                        if (hit.tFar < tNearest)
                        {
                            tNearest = hit.tFar;
                            hitIdx = idx;
                        }
                    }
                }
            };
            bih.TestRay(ray, onIntersect);

            if (hit.tFar != kFltMax)
            {
                L += segments[hitIdx].GetColour();
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
        const vec2 xyView = m_params.view.matrix * vec2(xyScreen);

        if (!m_params.sceneBounds.Contains(xyView)) 
        { 
            deviceOutputImage->At(xyScreen) = vec4(0.1f, 0.1f, 0.1f, 1.0f);
            return; 
        }

        const vec2 uv = vec2(xyScreen) * vec2(m_objects.accumBuffer->Dimensions()) / vec2(deviceOutputImage->Dimensions());
        vec4 L = m_objects.accumBuffer->Lerp(uv);
        L.xyz /= max(L.w, 1.0f);

        deviceOutputImage->At(xyScreen) = vec4(L.xyz, 1.0f);
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    Host::PathTracer::PathTracer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<Cuda::Host::Vector<GI2D::LineSegment>>& lineSegments, 
                                         const uint width, const uint height, const uint downsample, cudaStream_t renderStream) :
        UILayer(id),
        m_hostBIH2D(bih),
        m_hostLineSegments(lineSegments)
    {
        // Create some Cuda objects
        m_hostAccumBuffer = Cuda::CreateAsset<Cuda::Host::ImageRGBW>("id_2dgiAccumBuffer", width / downsample, height / downsample, renderStream);

        m_objects.bih = m_hostBIH2D->GetDeviceInstance();
        m_objects.lineSegments = m_hostLineSegments->GetDeviceInstance();
        m_objects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();

        m_params.accum.width = width;
        m_params.accum.height = height;
        m_params.accum.downsample = downsample;

        cu_deviceData = InstantiateOnDevice<Device::PathTracer>(GetAssetID(), m_params, m_objects);
    }

    Host::PathTracer::~PathTracer()
    {
        OnDestroyAsset();
    }

    __host__ void Host::PathTracer::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceData);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::PathTracer::Render()
    {
        if (m_isDirty)
        {
            m_hostAccumBuffer->Clear(vec4(0.0f));
            m_isDirty = false;
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
    
    __host__ void Host::PathTracer::Synchronise()
    {
        if (!m_isDirty) { return; }
        
        m_params.view = m_viewCtx.transform;

        SynchroniseObjects(cu_deviceData, m_params);
        m_isDirty = false;        
    }
}