#include "VoxelProxyGrid.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/Hash.h"
#include "core/CudaHeaders.cuh"

#include "../RenderCtx.cuh"
#include "../scene/SceneContainer.cuh"
#include "core/2d/primitives/SDF.cuh"
#include "../tracables/Tracable.cuh"

namespace Enso
{
    __device__ bool Device::VoxelProxyGrid::CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const
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

    __device__ void Device::VoxelProxyGrid::Accumulate(const vec4& L, const RenderCtx& ctx)
    {
        const int accumIdx = kKernelIdx * Camera::m_params.accum.numHarmonics;
        for (int harIdx = 0; harIdx < Camera::m_params.accum.numHarmonics; ++harIdx)
        {
            (*m_objects.accumBuffer)[accumIdx + harIdx] += L.xyz;
        }
    }

    __device__ vec3 Device::VoxelProxyGrid::Evaluate(const vec2& posWorld) const
    {
        const ivec2 probeIdx = ivec2(m_params.cameraTransform.PointToObjectSpace(posWorld));

        if (probeIdx.x < 0 || probeIdx.x >= m_params.gridSize.x || probeIdx.y < 0 || probeIdx.y >= m_params.gridSize.y) { return kOne * 0.2; }

        return Camera::m_objects.accumBuffer->Evaluate(probeIdx.y * m_params.gridSize.x + probeIdx.x, 0);
    }

    __host__ __device__ vec4 Device::VoxelProxyGrid::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
#ifdef __CUDA_ARCH__
        return vec4(Evaluate(pWorld), 1.f);
#else
        return vec4(0.f); 
#endif
    }

    Host::VoxelProxyGrid::VoxelProxyGrid(const Asset::InitCtx& initCtx, const Json::Node& json, const AssetHandle<const Host::SceneContainer>& scene) :
        Camera(initCtx, &m_hostInstance, scene),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::VoxelProxyGrid>(*this))
    {
        m_hostInstance.SceneObject::m_params.worldBBox = m_hostInstance.SceneObject::m_params.objectBBox + m_hostInstance.SceneObject::m_params.transform.trans;

        Camera::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Camera>(cu_deviceInstance));

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

    __host__ bool Host::VoxelProxyGrid::Serialise(Json::Node& node, const int flags) const
    {
        Camera::Serialise(node, flags);        
        return true;
    }

    __host__ bool Host::VoxelProxyGrid::Deserialise(const Json::Node& node, const int flags)
    {
        return Camera::Deserialise(node, flags);
    }

    /*void Host::VoxelProxyGrid::Bind()
    {
        m_scene = scene;
        m_deviceObjects.scene = m_scene->GetDeviceInstance();

        Synchronise(kSyncParams | kSyncObjects);
    }*/

    Host::VoxelProxyGrid::~VoxelProxyGrid() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    /*__host__ AssetHandle<Host::GenericObject> Host::VoxelProxyGrid::Instantiate(const std::string& id, const Json::Node& json, const AssetHandle<const Host::SceneContainer>& scene)
    {
        return AssetAllocator::CreateAsset<Host::VoxelProxyGrid>(id, json, scene);
    }*/

    __host__ void Host::VoxelProxyGrid::OnSynchroniseCamera(const uint syncFlags)
    {        
        if (syncFlags & kSyncObjects) { SynchroniseObjects<Device::VoxelProxyGrid>(cu_deviceInstance, m_deviceObjects); }
        if (syncFlags & kSyncParams) { SynchroniseObjects<Device::VoxelProxyGrid>(cu_deviceInstance, m_params); }
    }
}