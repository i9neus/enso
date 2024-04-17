#include "VoxelProxyGridLayer.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/Hash.h"
#include "core/CudaHeaders.cuh"

#include "../RenderCtx.cuh"
#include "../SceneDescription.cuh"
#include "../primitives/SDF.cuh"

namespace Enso
{
    __device__ bool Device::VoxelProxyGridLayer::CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const
    {
        const uint probeIdx = kKernelIdx / Camera::m_params.accum.subprobesPerProbe;
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

    __device__ void Device::VoxelProxyGridLayer::Accumulate(const vec4& L, const RenderCtx& ctx)
    {
        const int accumIdx = kKernelIdx * Camera::m_params.accum.numHarmonics;
        for (int harIdx = 0; harIdx < Camera::m_params.accum.numHarmonics; ++harIdx)
        {
            (*m_objects.accumBuffer)[accumIdx + harIdx] += L.xyz;
        }
    }

    __device__ vec3 Device::VoxelProxyGridLayer::Evaluate(const vec2& posWorld) const
    {
        const ivec2 probeIdx = ivec2(m_params.cameraTransform.PointToObjectSpace(posWorld));

        if (probeIdx.x < 0 || probeIdx.x >= m_params.gridSize.x || probeIdx.y < 0 || probeIdx.y >= m_params.gridSize.y) { return kOne * 0.2; }

        return Camera::m_objects.accumBuffer->Evaluate(probeIdx.y * m_params.gridSize.x + probeIdx.x, 0);
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
        Camera(id, m_hostInstance, scene),
        cu_deviceInstance(m_allocator.InstantiateOnDevice<Device::VoxelProxyGridLayer>())
    {
        Camera::SetDeviceInstance(m_allocator.StaticCastOnDevice<Device::Camera>(cu_deviceInstance));

        constexpr uint kGridWidth = 100;
        constexpr uint kGridHeight = 100;
        constexpr uint kNumHarmonics = 1;
        constexpr size_t kAccumBufferSize = 1024 * 1024;        

        Camera::Initialise(kGridWidth * kGridHeight, kNumHarmonics, kAccumBufferSize);

        // Construct the camera transform
        m_params.cameraTransform.Construct(vec2(-0.5f), 0.0f, float(kGridWidth));      

        // Set the device objects
        m_deviceObjects.accumBuffer = Camera::m_accumBuffer->GetDeviceInstance();

        // Cache some parameters used for the accumulator
        m_params.gridSize = ivec2(kGridWidth, kGridHeight);

        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ bool Host::VoxelProxyGridLayer::Serialise(Json::Node& node, const int flags) const
    {
        Camera::Serialise(node, flags);        
        return true;
    }

    __host__ uint Host::VoxelProxyGridLayer::Deserialise(const Json::Node& node, const int flags)
    {
        GenericObject::m_dirtyFlags |= Camera::Deserialise(node, flags);

        return GenericObject::m_dirtyFlags;
    }

    /*void Host::VoxelProxyGridLayer::Bind()
    {
        m_scene = scene;
        m_deviceObjects.scene = m_scene->GetDeviceInstance();

        Synchronise(kSyncParams | kSyncObjects);
    }*/

    Host::VoxelProxyGridLayer::~VoxelProxyGridLayer()
    {
        m_allocator.DestroyOnDevice(cu_deviceInstance);
    }

    /*__host__ AssetHandle<Host::GenericObject> Host::VoxelProxyGridLayer::Instantiate(const std::string& id, const Json::Node& json, const AssetHandle<const Host::SceneDescription>& scene)
    {
        return CreateAsset<Host::VoxelProxyGridLayer>(id, json, scene);
    }*/

    __host__ void Host::VoxelProxyGridLayer::Render()
    {
    }

    __host__ bool Host::VoxelProxyGridLayer::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {           
        Camera::Rebuild(dirtyFlags, viewCtx);
        
        m_params.viewCtx = viewCtx;

        Synchronise(kSyncParams);

        return true;
    }

    __host__ void Host::VoxelProxyGridLayer::Synchronise(const int syncType)
    {
        Camera::Synchronise(syncType);
        
        if (syncType & kSyncObjects) { SynchroniseObjects<Device::VoxelProxyGridLayer>(cu_deviceInstance, m_deviceObjects); }
        if (syncType & kSyncParams) { SynchroniseObjects<Device::VoxelProxyGridLayer>(cu_deviceInstance, m_params); }
    }

    __host__ void Host::VoxelProxyGridLayer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(hostOutputImage, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0 >> > (cu_deviceInstance, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }
}