#include "CudaGI2DPathTracer.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"

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
        //const vec2 xyView = m_params.view.matrix * vec2(xyScreen);

        //if (!m_params.sceneBounds.Contains(xyView)) { return; }

        vec3 colour1 = Hue(m_params.frameIdx / 60.0f), colour2 = Hue(0.5f + m_params.frameIdx / 60.0f);
        m_objects.accumBuffer->At(xyScreen) = vec4(((xyScreen.x / 10 + xyScreen.y / 10) % 2 == 0) ? colour1 : colour2, 1.0f);
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
            deviceOutputImage->At(xyScreen) = vec4(0.0f, 0.0f, 0.0f, 1.0f);
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
        Asset(id),
        m_hostBIH2D(bih),
        m_hostLineSegments(lineSegments)
    {
        // Create some Cuda objects
        m_hostAccumBuffer = Cuda::CreateAsset<Cuda::Host::ImageRGBW>("id_2dgiAccumBuffer", width / downsample, height / downsample, renderStream);
        
        m_objects.bih = m_hostBIH2D->GetDeviceInstance();
        m_objects.lineSegments = m_hostLineSegments->GetDeviceInstance();
        m_objects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();

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

    template<typename T>
    __host__ void KernelParamsFromImage(const AssetHandle<Cuda::Host::Image<T>>& image, dim3& blockSize, dim3& gridSize)
    {
        const auto& meta = image->GetMetadata();
        blockSize = dim3(16, 16, 1);
        gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
    }

    __host__ void Host::PathTracer::Render()
    {        
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelRender << < gridSize, blockSize, 0 >> > (cu_deviceData);
        IsOk(cudaDeviceSynchronize());
    }
    
    __host__ void Host::PathTracer::Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage)
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(hostOutputImage, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0 >> > (cu_deviceData, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::PathTracer::SetParams(const PathTracerParams& newParams)
    {
        m_params.view = newParams.view;
        m_params.sceneBounds = newParams.sceneBounds;
        m_params.frameIdx = newParams.frameIdx;

        SynchroniseObjects(cu_deviceData, m_params);
    }
}