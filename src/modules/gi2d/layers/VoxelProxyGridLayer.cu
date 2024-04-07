#include "VoxelProxyGridLayer.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/Hash.h"
#include "core/CudaHeaders.cuh"

#include "../RenderCtx.cuh"
#include "../SceneDescription.cuh"
#include "../primitives/SDF.cuh"

namespace Enso
{
    constexpr size_t kAccumBufferSize = 1024 * 1024;
    
    /*__host__ __device__ void Device::VoxelProxyGridLayer::OnSynchronise(const int syncFlags)
    {
        if (syncFlags == kSyncObjects)
        {
            m_scene = *m_objects.scene;
        }
    }*/

    __device__ void Device::VoxelProxyGridLayer::Synchronise(const VoxelProxyGridLayerObjects& objects)
    {
        m_objects = objects;
        Camera2D::Synchronise(*m_objects.scene);
    }

    __device__ void Device::VoxelProxyGridLayer::Accumulate(const vec4& L, const RenderCtx& ctx)
    {       
        int accumIdx = kKernelIdx * m_params.accum.numHarmonics;  

        for (int harIdx = 0; harIdx < m_params.accum.numHarmonics; ++harIdx)
        {
            (*m_objects.accumBuffer)[accumIdx + harIdx] += L.xyz; 
        }
    }

    __device__ bool Device::VoxelProxyGridLayer::CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const
    {
        const uint probeIdx = kKernelIdx / m_params.accum.subprobesPerProbe;
        const vec2 probePosNorm = vec2(float(0.5f + probeIdx % m_params.gridSize.x), float(0.5f + probeIdx / m_params.gridSize.x));

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

    DEFINE_KERNEL_PASSTHROUGH_ARGS(Prepare);

    __device__ void Device::VoxelProxyGridLayer::Render()
    {
        if (kKernelIdx >= m_params.accum.totalSubprobes) { return; }
        CudaAssertDebug(m_objects.accumBuffer)

        const int subprobeIdx = ((kKernelIdx / m_params.accum.numHarmonics) % m_params.accum.subprobesPerProbe);
        const int probeIdx = kKernelIdx / m_params.accum.unitsPerProbe;

        const uchar ctxFlags = (probeIdx / m_params.gridSize.x == m_params.gridSize.y / 2 &&
            probeIdx % m_params.gridSize.x == m_params.gridSize.x / 2 &&
            subprobeIdx == 0) ? kRenderCtxDebug : 0;

        Integrate(ctxFlags);
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ vec3 Device::VoxelProxyGridLayer::Evaluate(const vec2& posWorld) const
    {
        const ivec2 probeIdx = ivec2(m_params.cameraTransform.PointToObjectSpace(posWorld));

        if (probeIdx.x < 0 || probeIdx.x >= m_params.gridSize.x || probeIdx.y < 0 || probeIdx.y >= m_params.gridSize.y) { return kOne * 0.2; }

        vec3 L = m_objects.accumBuffer->Evaluate(probeIdx.y * m_params.gridSize.x + probeIdx.x, 0);

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
        CudaAssertDebug(deviceOutputImage);

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

        m_accumBuffer = CreateChildAsset<Host::AccumulationBuffer>("accumBuffer", kGridWidth * kGridHeight, kNumHarmonics);

        // Construct the camera transform
        m_params.cameraTransform.Construct(vec2(-0.5f), 0.0f, float(kGridWidth));      

        // Set the device objects
        m_deviceObjects.scene = m_scene->GetDeviceInstance();
        m_deviceObjects.accumBuffer = m_accumBuffer->GetDeviceInstance();

        // Cache some parameters used for the accumulator
        m_params.gridSize = ivec2(kGridWidth, kGridHeight);
        m_params.accum = m_accumBuffer->GetParams();

        // Instantiate and sync
        cu_deviceInstance = InstantiateOnDevice<Device::VoxelProxyGridLayer>();
        Synchronise(kSyncParams | kSyncObjects);
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
        m_deviceObjects.scene = m_scene->GetDeviceInstance();

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
    }

    __host__ void Host::VoxelProxyGridLayer::Render()
    {
        KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);
        
        if (m_dirtyFlags & kDirtyIntegrators)
        {
            m_accumBuffer->Clear();
        }
        m_dirtyFlags = 0;

        ScopedDeviceStackResize(1024 * 10, [this]() -> void
            {
                KernelRender << <m_params.accum.kernel.grids.accumSize, m_params.accum.kernel.blockSize >> > (cu_deviceInstance);
            });

        m_accumBuffer->Reduce();

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