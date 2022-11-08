#include "VoxelProxyGridLayer.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "generic/Hash.h"

#include "../RenderCtx.cuh"
#include "../SceneDescription.cuh"
#include "../integrators/VoxelProxyGrid.cuh"

using namespace Cuda;

namespace GI2D
{
    constexpr size_t kAccumBufferSize = 1024 * 1024;
    
    __host__ __device__ VoxelProxyGridLayerParams::VoxelProxyGridLayerParams()
    {

    }

    __device__ void Device::VoxelProxyGridLayer::OnSynchronise(const int syncFlags)
    {
        if (syncFlags == kSyncObjects)
        {
            assert(m_scenePtr);
            m_scene = *m_scenePtr;
        }
    }

    __device__ void Device::VoxelProxyGridLayer::Accumulate(const vec4& L, const RenderCtx& ctx)
    {
        int accumIdx = kKernelIdx * m_grid.numHarmonics;

        for (int harIdx = 0; harIdx < m_grid.numHarmonics; ++harIdx)
        {
            (*m_accumBuffer)[accumIdx + harIdx] += L.xyz;
        }
    }

    __device__ bool Device::VoxelProxyGridLayer::CreateRay(Ray2D& ray, RenderCtx& renderCtx) const
    {
        const uint probeIdx = kKernelIdx / m_grid.subprobesPerProbe;
        const vec2 probePosNorm = vec2(float(0.5f + probeIdx % m_grid.size.x), float(0.5f + probeIdx / m_grid.size.x));

        // Transform from screen space to view space
        ray.o = m_cameraTransform.PointToWorldSpace(probePosNorm);

        // Randomly scatter
        const float theta = renderCtx.rng.Rand<0>() * kTwoPi;
        ray.d = vec2(cosf(theta), sinf(theta));

        return true;
    }

    __device__ void Device::VoxelProxyGridLayer::Render()
    {
        if (kKernelIdx >= m_grid.totalSubprobes) { return; }

        RenderCtx renderCtx(kKernelIdx, uint(m_frameIdx), 0, *this);
        m_voxelTracer.Integrate(renderCtx);

    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::VoxelProxyGridLayer::ReduceAccumulationBuffer(const uint batchSize, const uvec2 batchRange)
    {
        if (kKernelIdx >= m_grid.totalAccumUnits) { return; }

        assert(m_reduceBuffer);
        assert(m_accumBuffer);

        auto& accumBuffer = *m_accumBuffer;
        auto& reduceBuffer = *m_reduceBuffer;
        const int subprobeIdx = (kKernelIdx / m_grid.numHarmonics) % m_grid.subprobesPerProbe; 

        for (uint iterationSize = batchRange[0] / 2; iterationSize > batchRange[1] / 2; iterationSize >>= 1)
        {
            if (subprobeIdx < iterationSize)
            {
                // For the first iteration, copy the data out of the accumulation buffer
                if (iterationSize == batchSize / 2)
                {
                    auto& texel = reduceBuffer[kKernelIdx];
                    texel = accumBuffer[kKernelIdx];

                    if (subprobeIdx + iterationSize < m_grid.subprobesPerProbe)
                    {
                        assert(kKernelIdx + iterationSize * m_grid.numHarmonics < kAccumBufferSize);                       
                        texel += accumBuffer[kKernelIdx + iterationSize * m_grid.numHarmonics];
                    }                 
                }
                else
                {
                    assert(kKernelIdx + iterationSize * m_grid.numHarmonics < kAccumBufferSize);
                    assert(subprobeIdx + iterationSize < m_grid.subprobesPerProbe);

                    reduceBuffer[kKernelIdx] += reduceBuffer[kKernelIdx + iterationSize * m_grid.numHarmonics];
                }
            }

            __syncthreads();
        }

        // After the last operation, cache the accumulated value in the probe grid
        if (subprobeIdx == 0 && batchRange[0] == 2)
        {
            const int probeIdx = kKernelIdx / m_grid.unitsPerProbe;
            const int coeffIdx = kKernelIdx % m_grid.numHarmonics;

            assert(probeIdx < m_grid.numProbes && coeffIdx < m_grid.numHarmonics);

            (*m_gridBuffer)[probeIdx * m_grid.numHarmonics + coeffIdx] = reduceBuffer[kKernelIdx];          
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(ReduceAccumulationBuffer);

    __device__ void Device::VoxelProxyGridLayer::Prepare(const uint dirtyFlags)
    {
        if (dirtyFlags)
        {
            // Save ourselves a deference here by caching the scene pointers
            assert(m_scenePtr);
            m_scene = *m_scenePtr;
            m_frameIdx = 0;
        }
        else
        {
            m_frameIdx++;
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Prepare);

    __device__ vec3 Device::VoxelProxyGridLayer::Evaluate(const vec2& posWorld) const
    {
        const ivec2 probeIdx = ivec2(m_cameraTransform.PointToObjectSpace(posWorld));

        if (probeIdx.x < 0 || probeIdx.x >= m_grid.size.x || probeIdx.y < 0 || probeIdx.y >= m_grid.size.y) { return kOne * 0.2; }

        return (*m_gridBuffer)[(probeIdx.y * m_grid.size.x + probeIdx.x) * m_grid.numHarmonics] / float(max(1, m_frameIdx * m_grid.subprobesPerProbe));
    }

    __device__ void Device::VoxelProxyGridLayer::Composite(Cuda::Device::ImageRGBA* deviceOutputImage)  const
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

        deviceOutputImage->At(xyScreen) = vec4(Evaluate(xyView), 1.0f);
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    Host::VoxelProxyGridLayer::VoxelProxyGridLayer(const std::string& id, AssetHandle<Host::SceneDescription>& scene,
                                                  const uint width, const uint height) :
        UILayer(id, scene)
    {
        Assert(m_scene);
        
        constexpr uint kGridWidth = 100;
        constexpr uint kGridHeight = 100;
        constexpr uint kNumHarmonics = 1;

        // Establish the properties of the grid
        m_grid.size = ivec2(kGridWidth, kGridHeight);
        m_grid.numProbes = Area(m_grid.size);        
        m_grid.numHarmonics = (kNumHarmonics - 1) * 2 + 1;
        m_grid.totalGridUnits = m_grid.numProbes * m_grid.numHarmonics;

        // Derive some more properties used when accumulating and reducing.
        m_grid.subprobesPerProbe = std::min(kAccumBufferSize / m_grid.numProbes,
                                            kAccumBufferSize / m_grid.totalGridUnits);
        m_grid.unitsPerProbe = m_grid.subprobesPerProbe * m_grid.numHarmonics;
        m_grid.totalSubprobes = m_grid.subprobesPerProbe * m_grid.numProbes;
        m_grid.totalAccumUnits = m_grid.totalSubprobes * m_grid.numHarmonics;

        // Construct the camera transform
        m_cameraTransform.Construct(vec2(-0.5f), 0.0f, float(kGridWidth));

        // Create some assets
        m_hostAccumBuffer = CreateChildAsset<Core::Host::Vector<vec3>>("accumBuffer", kAccumBufferSize, Core::kVectorHostAlloc);
        m_hostReduceBuffer = CreateChildAsset<Core::Host::Vector<vec3>>("reduceBuffer", kAccumBufferSize, Core::kVectorHostAlloc);
        m_hostProxyGrid = CreateChildAsset<Core::Host::Vector<vec3>>("proxyGrid", m_grid.totalGridUnits, Core::kVectorHostAlloc);

        // Set the device objects
        m_deviceObjects.m_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.m_reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
        m_deviceObjects.m_gridBuffer = m_hostProxyGrid->GetDeviceInstance();
        m_deviceObjects.m_scenePtr = m_scene->GetDeviceInstance();

        // Instantiate and sync
        cu_deviceInstance = InstantiateOnDevice<Device::VoxelProxyGridLayer>();
        Synchronise(kSyncParams | kSyncObjects);

        // Set the parameters for the accumulate and reduce kernels
        m_kernelParams.blockSize = 256;
        m_kernelParams.grids.accumSize = (m_grid.totalSubprobes + m_kernelParams.blockSize - 1) / m_kernelParams.blockSize;
        m_kernelParams.grids.reduceSize = (m_grid.totalAccumUnits + m_kernelParams.blockSize - 1) / m_kernelParams.blockSize;
    }

    Host::VoxelProxyGridLayer::~VoxelProxyGridLayer()
    {
        OnDestroyAsset();
    }

    __host__ void Host::VoxelProxyGridLayer::Integrate()
    {
        // Used when parallel reducing the accumluation buffer
        const uint reduceBatchSizePow2 = NearestPow2Ceil(m_grid.subprobesPerProbe);

        // Reduce until the batch range is equal to the size of the block
        uint batchSize = reduceBatchSizePow2;
        while (batchSize > 1)
        {
            KernelReduceAccumulationBuffer << < m_kernelParams.grids.reduceSize, m_kernelParams.blockSize>> > ( cu_deviceInstance, reduceBatchSizePow2, uvec2(batchSize, batchSize >> 1));
            batchSize >>= 1;
        }
        IsOk(cudaDeviceSynchronize());     
    }

    __host__ void Host::VoxelProxyGridLayer::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {
        UILayer::Rebuild(dirtyFlags, viewCtx, selectionCtx);

        Synchronise(kSyncParams);
    }

    __host__ void Host::VoxelProxyGridLayer::Synchronise(const int syncType)
    {
        UILayer::Synchronise(cu_deviceInstance, syncType);

        if (syncType & kSyncObjects) { SynchroniseInheritedClass<VoxelProxyGridLayerObjects>(cu_deviceInstance, m_deviceObjects, kSyncObjects); }
        if (syncType & kSyncParams) { SynchroniseInheritedClass<VoxelProxyGridLayerParams>(cu_deviceInstance, *this, kSyncParams); }
    }

    __host__ void Host::VoxelProxyGridLayer::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceInstance);

        m_hostAccumBuffer.DestroyAsset();
        m_hostReduceBuffer.DestroyAsset();
        m_hostProxyGrid.DestroyAsset();
    }

    __host__ void Host::VoxelProxyGridLayer::Render()
    {
        KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        if (m_dirtyFlags)
        {
            m_hostAccumBuffer->Wipe();
            m_dirtyFlags = 0;
        }

        const int blockSize = 16;
        const int gridSize = (m_grid.numProbes + blockSize - 1) / blockSize;
        
        KernelRender << <m_kernelParams.grids.accumSize, m_kernelParams.blockSize>> > (cu_deviceInstance);

        Integrate();

        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::VoxelProxyGridLayer::Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(hostOutputImage, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0 >> > (cu_deviceInstance, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }
}