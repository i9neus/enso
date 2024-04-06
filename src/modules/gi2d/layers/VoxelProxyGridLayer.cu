#include "VoxelProxyGridLayer.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/Hash.h"
#include "core/CudaHeaders.cuh"

#include "../RenderCtx.cuh"
#include "../SceneDescription.cuh"
#include "../integrators/VoxelProxyGrid.cuh"
#include "../primitives/SDF.cuh"

namespace Enso
{
    constexpr size_t kAccumBufferSize = 1024 * 1024;
    
    __host__ __device__ VoxelProxyGridLayerParams::VoxelProxyGridLayerParams()
    {

    }

    __host__ __device__ void Device::VoxelProxyGridLayer::OnSynchronise(const int syncFlags)
    {
        if (syncFlags == kSyncObjects)
        {
            m_scene = *m_objects.scenePtr;
        }
    }

    __device__ void Device::VoxelProxyGridLayer::Accumulate(const vec4& L, const RenderCtx& ctx)
    {
        int accumIdx = kKernelIdx * m_params.grid.numHarmonics;

        for (int harIdx = 0; harIdx < m_params.grid.numHarmonics; ++harIdx)
        {
            (*m_objects.accumBuffer)[accumIdx + harIdx] += L.xyz; 
        }
    }

    __device__ bool Device::VoxelProxyGridLayer::CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const
    {
        const uint probeIdx = kKernelIdx / m_params.grid.subprobesPerProbe;
        const vec2 probePosNorm = vec2(float(0.5f + probeIdx % m_params.grid.size.x), float(0.5f + probeIdx / m_params.grid.size.x));

        // Transform from screen space to view space
        ray.o = m_params.cameraTransform.PointToWorldSpace(probePosNorm);

        // Randomly scatter the ray
        const float theta = renderCtx.rng.Rand<0>() * kTwoPi;
        ray.d = vec2(cosf(theta), sinf(theta));

        /*if (renderCtx.IsDebug())
        {
            ray.o = vec2(0.0f);
            ray.d = normalize(UILayer::m_params.viewCtx.mousePos - ray.o);
        }*/

        ray.throughput = vec3(1.0f);
        ray.flags = 0;
        ray.lightIdx = kTracableNotALight;

        // Initialise the hit context
        hit.flags = kHit2DIsVolume;
        hit.p = ray.o;
        hit.tFar = kFltMax;
        hit.depth = 0;

        return true;
    }

    __device__ void Device::VoxelProxyGridLayer::Render()
    {
        if (kKernelIdx >= m_params.grid.totalSubprobes) { return; }
         
        const int subprobeIdx = ((kKernelIdx / m_params.grid.numHarmonics) % m_params.grid.subprobesPerProbe);
        const int probeIdx = kKernelIdx / m_params.grid.unitsPerProbe;

        const uchar ctxFlags = (probeIdx / m_params.grid.size.x == m_params.grid.size.y / 2 && 
                               probeIdx % m_params.grid.size.x == m_params.grid.size.x / 2 &&
                               subprobeIdx == 0) ? kRenderCtxDebug : 0;
        
        RenderCtx renderCtx(kKernelIdx, uint(m_frameIdx), 0, *this, ctxFlags);
        /*if (ctxFlags & kRenderCtxDebug)
        {
            renderCtx.debugData = &m_kifsDebug;
        }*/

        m_voxelTracer.Integrate(renderCtx);
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::VoxelProxyGridLayer::ReduceAccumulationBuffer(const uint batchSize, const uvec2 batchRange)
    {
        if (kKernelIdx >= m_params.grid.totalAccumUnits) { return; }

        assert(m_reduceBuffer);
        assert(m_accumBuffer);

        auto& accumBuffer = *m_objects.accumBuffer;
        auto& reduceBuffer = *m_objects.reduceBuffer;
        const int subprobeIdx = (kKernelIdx / m_params.grid.numHarmonics) % m_params.grid.subprobesPerProbe; 

        for (uint iterationSize = batchRange[0] / 2; iterationSize > batchRange[1] / 2; iterationSize >>= 1)
        {
            if (subprobeIdx < iterationSize)
            {
                // For the first iteration, copy the data out of the accumulation buffer
                if (iterationSize == batchSize / 2)
                {
                    auto& texel = reduceBuffer[kKernelIdx];
                    texel = accumBuffer[kKernelIdx];

                    if (subprobeIdx + iterationSize < m_params.grid.subprobesPerProbe)
                    {
                        assert(kKernelIdx + iterationSize * m_params.grid.numHarmonics < kAccumBufferSize);                       
                        texel += accumBuffer[kKernelIdx + iterationSize * m_params.grid.numHarmonics];
                    }                 
                }
                else
                {
                    assert(kKernelIdx + iterationSize * m_params.grid.numHarmonics < kAccumBufferSize);
                    assert(subprobeIdx + iterationSize < m_params.grid.subprobesPerProbe);

                    reduceBuffer[kKernelIdx] += reduceBuffer[kKernelIdx + iterationSize * m_params.grid.numHarmonics];
                }
            }

            __syncthreads();
        }

        // After the last operation, cache the accumulated value in the probe grid
        if (subprobeIdx == 0 && batchRange[0] == 2)
        {
            const int probeIdx = kKernelIdx / m_params.grid.unitsPerProbe;
            const int coeffIdx = kKernelIdx % m_params.grid.numHarmonics;

            assert(probeIdx < m_params.grid.numProbes && coeffIdx < m_params.grid.numHarmonics);

            (*m_objects.gridBuffer)[probeIdx * m_params.grid.numHarmonics + coeffIdx] = reduceBuffer[kKernelIdx];
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(ReduceAccumulationBuffer);

    __device__ void Device::VoxelProxyGridLayer::Prepare(const uint dirtyFlags)
    {
        if (dirtyFlags & kDirtyIntegrators)
        {
            // Save ourselves a dereference here by caching the scene pointers
            assert(m_objects.scenePtr);
            m_scene = *m_objects.scenePtr;
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
        const ivec2 probeIdx = ivec2(m_params.cameraTransform.PointToObjectSpace(posWorld));

        if (probeIdx.x < 0 || probeIdx.x >= m_params.grid.size.x || probeIdx.y < 0 || probeIdx.y >= m_params.grid.size.y) { return kOne * 0.2; }

        vec3 L = (*m_objects.gridBuffer)[(probeIdx.y * m_params.grid.size.x + probeIdx.x) * m_params.grid.numHarmonics] / float(m_params.grid.subprobesPerProbe * max(1, m_frameIdx));

        /*if (length(posWorld - m_kifsDebug.pNear) < UILayer::m_params.viewCtx.dPdXY * 4.0f) { L += kRed; }
        if (length(posWorld - m_kifsDebug.pFar) < UILayer::m_params.viewCtx.dPdXY * 4.0f) { L += kYellow; }
        for (int idx = 0; idx < KIFSDebugData::kMaxPoints; ++idx)
        {
            if (length(posWorld - m_kifsDebug.marchPts[idx]) < UILayer::m_params.viewCtx.dPdXY * 4.0f) { L += kGreen; }
        }*/

        /*if (m_kifsDebug.isHit)
        {
            if (length(posWorld - m_kifsDebug.hit) < UILayer::m_params.viewCtx.dPdXY * 5.0f) { L += kRed; }
            if (SDFLine(posWorld, m_kifsDebug.hit, m_kifsDebug.normal * UILayer::m_params.viewCtx.dPdXY * 100.0f).x < UILayer::m_params.viewCtx.dPdXY * 1.f) { L += kRed; }
        }*/

        return L;
    }

    __device__ void Device::VoxelProxyGridLayer::Composite(Device::ImageRGBA* deviceOutputImage)  const
    {
        assert(deviceOutputImage);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= deviceOutputImage->Width() || xyScreen.y < 0 || xyScreen.y >= deviceOutputImage->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_params.viewCtx.transform.matrix * vec2(xyScreen);

        if (!m_params.viewCtx.sceneBounds.Contains(xyView))
        {
            deviceOutputImage->At(xyScreen) = vec4(0.1f, 0.1f, 0.1f, 1.0f);
            return;
        }

        deviceOutputImage->At(xyScreen) = vec4(Evaluate(xyView), 1.0f);
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    Host::VoxelProxyGridLayer::VoxelProxyGridLayer(const std::string& id, const Json::Node& json, const AssetHandle<const Host::SceneDescription>& scene) :
        GenericObject(id),
        m_scene(scene)
    {
        Assert(m_scene);

        constexpr uint kGridWidth = 100;
        constexpr uint kGridHeight = 100;
        constexpr uint kNumHarmonics = 1;

        // Establish the properties of the grid
        m_params.grid.size = ivec2(kGridWidth, kGridHeight);
        m_params.grid.numProbes = Area(m_params.grid.size);
        m_params.grid.numHarmonics = (kNumHarmonics - 1) * 2 + 1;
        m_params.grid.totalGridUnits = m_params.grid.numProbes * m_params.grid.numHarmonics;

        // Derive some more properties used when accumulating and reducing.
        m_params.grid.subprobesPerProbe = std::min(kAccumBufferSize / m_params.grid.numProbes,
            kAccumBufferSize / m_params.grid.totalGridUnits);
        m_params.grid.unitsPerProbe = m_params.grid.subprobesPerProbe * m_params.grid.numHarmonics;
        m_params.grid.totalSubprobes = m_params.grid.subprobesPerProbe * m_params.grid.numProbes;
        m_params.grid.totalAccumUnits = m_params.grid.totalSubprobes * m_params.grid.numHarmonics;

        // Construct the camera transform
        m_params.cameraTransform.Construct(vec2(-0.5f), 0.0f, float(kGridWidth));

        // Create some assets
        m_hostAccumBuffer = CreateChildAsset<Host::Vector<vec3>>("accumBuffer", kAccumBufferSize, kVectorHostAlloc);
        m_hostReduceBuffer = CreateChildAsset<Host::Vector<vec3>>("reduceBuffer", kAccumBufferSize, kVectorHostAlloc);
        m_hostProxyGrid = CreateChildAsset<Host::Vector<vec3>>("proxyGrid", m_params.grid.totalGridUnits, kVectorHostAlloc);

        // Set the device objects
        m_deviceObjects.accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
        m_deviceObjects.gridBuffer = m_hostProxyGrid->GetDeviceInstance();
        m_deviceObjects.scenePtr = m_scene->GetDeviceInstance();

        // Instantiate and sync
        cu_deviceInstance = InstantiateOnDevice<Device::VoxelProxyGridLayer>();
        Synchronise(kSyncParams | kSyncObjects);

        // Set the parameters for the accumulate and reduce kernels
        m_kernelParams.blockSize = 256;
        m_kernelParams.grids.accumSize = (m_params.grid.totalSubprobes + m_kernelParams.blockSize - 1) / m_kernelParams.blockSize;
        m_kernelParams.grids.reduceSize = (m_params.grid.totalAccumUnits + m_kernelParams.blockSize - 1) / m_kernelParams.blockSize;
    }

    __host__ bool Host::VoxelProxyGridLayer::Serialise(Json::Node& node, const int flags) const
    {
        
        return true;
    }

    __host__ uint Host::VoxelProxyGridLayer::Deserialise(const Json::Node& node, const int flags)
    {
       
        return m_dirtyFlags;
    }

    /*void Host::VoxelProxyGridLayer::Bind()
    {
        m_scene = scene;
        m_deviceObjects.scenePtr = m_scene->GetDeviceInstance();

        Synchronise(kSyncParams | kSyncObjects);
    }*/

    Host::VoxelProxyGridLayer::~VoxelProxyGridLayer()
    {
        OnDestroyAsset();
    }

    /*__host__ AssetHandle<Host::GenericObject> Host::VoxelProxyGridLayer::Instantiate(const std::string& id, const Json::Node& json, const AssetHandle<const Host::SceneDescription>& scene)
    {
        return CreateAsset<Host::VoxelProxyGridLayer>(id, json, scene);
    }*/

    __host__ void Host::VoxelProxyGridLayer::Reduce()
    {
        // Used when parallel reducing the accumluation buffer
        const uint reduceBatchSizePow2 = NearestPow2Ceil(m_params.grid.subprobesPerProbe);

        // Reduce until the batch range is equal to the size of the block
        uint batchSize = reduceBatchSizePow2;
        while (batchSize > 1)
        {
            KernelReduceAccumulationBuffer << < m_kernelParams.grids.reduceSize, m_kernelParams.blockSize >> > (cu_deviceInstance, reduceBatchSizePow2, uvec2(batchSize, batchSize >> 1));
            batchSize >>= 1;
        }
        IsOk(cudaDeviceSynchronize());
    }

    __host__ bool Host::VoxelProxyGridLayer::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx)
    {
        m_dirtyFlags = dirtyFlags;
        m_params.viewCtx = viewCtx;

        Synchronise(kSyncParams);

        return true;
    }

    __host__ void Host::VoxelProxyGridLayer::Synchronise(const int syncType)
    {
        if (syncType & kSyncObjects) { SynchroniseObjects<Device::VoxelProxyGridLayer>(cu_deviceInstance, m_deviceObjects); }
        if (syncType & kSyncParams) { SynchroniseObjects<Device::VoxelProxyGridLayer>(cu_deviceInstance, m_params); }
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

        if (m_dirtyFlags & kDirtyIntegrators)
        {
            m_hostAccumBuffer->Wipe();
        }
        m_dirtyFlags = 0;
        
        ScopedDeviceStackResize(1024 * 10, [this]() -> void
            {
                KernelRender << <m_kernelParams.grids.accumSize, m_kernelParams.blockSize >> > (cu_deviceInstance);
            });

        Reduce();

        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::VoxelProxyGridLayer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(hostOutputImage, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0 >> > (cu_deviceInstance, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }
}