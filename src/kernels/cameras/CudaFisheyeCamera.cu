#include "../CudaSampler.cuh"
#include "../CudaHash.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRay.cuh"

#include "CudaFisheyeCamera.cuh"
#include "generic/JsonUtils.h"
#include "../CudaManagedArray.cuh"
#include "../CudaManagedObject.cuh"

#include "../math/CudaSphericalHarmonics.cuh"

#define kCameraAA                 1.5f             // The width/height of the anti-aliasing kernel in pixels
#define kCameraSensorSize         0.035f           // The size of the camera sensor in meters
#define kBlades                   5.0f
#define kBladeCurvature           0.0f
#define kCameraUp                 vec3(0.0f, 1.0f, 0.0f) 

namespace Cuda
{
    __host__ __device__ FisheyeCameraParams::FisheyeCameraParams()
    {
    }

    __host__ FisheyeCameraParams::FisheyeCameraParams(const ::Json::Node& node) :
        FisheyeCameraParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void FisheyeCameraParams::ToJson(::Json::Node& node) const
    {
        camera.ToJson(node);
        transform.ToJson(node);
    }

    __host__ void FisheyeCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        camera.FromJson(node, flags);
        transform.FromJson(node, flags);
    }

    __device__ Device::FisheyeCamera::FisheyeCamera()
    {
        Prepare();
    }

    __device__ void Device::FisheyeCamera::Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const
    {
        if (viewportPos.x >= deviceOutputImage->Width() || viewportPos.y >= deviceOutputImage->Height() ||
            viewportPos.x >= m_objects.cu_accumBuffer->Width() || viewportPos.y >= m_objects.cu_accumBuffer->Height()) {
            return;
        }

        // If the texel weight is negative, the texel is ready to be rendered
        vec4& texel = m_objects.cu_accumBuffer->At(viewportPos);
        if (texel.w >= 0.0f) { return; }

        CompressedRay& compressedRay = (*m_objects.renderState.cu_compressedRayBuffer)[kKernelIdx];

        // Flip the weight back to positve
        texel.w = -texel.w;

        // Normalise and gamma correct
        vec3 rgb = pow(texel.xyz / fmax(1.0f, texel.w), vec3(1.0f / 1.0f));
        deviceOutputImage->At(viewportPos) = vec4(rgb, 1.0f);
    }

    __device__ void Device::FisheyeCamera::SeedRayBuffer(const ivec2& viewportPos)
    {
        CompressedRay& compressedRay = (*m_objects.renderState.cu_compressedRayBuffer)[viewportPos.y * 512 + viewportPos.x];

        if (!compressedRay.IsAlive())
        {
            /*if (m_params.isRealtime)
            {
                m_objects.cu_accumBuffer->At(compressedRay.GetViewportPos()) = vec4(0.0f);
            }*/

            CreateRay(viewportPos, compressedRay);
        }
    }

    __device__ void Device::FisheyeCamera::Prepare()
    {

    }

    __device__ bool Device::FisheyeCamera::SphericalViewportToCartesian(const ivec2& viewportPos, vec3& cart) const
    {
        const vec2 uv = vec2(viewportPos - 256) / 256.0f;
        if (length(uv) >= 1.0f) { return false; }

        cart = PolarToCartesian(vec2(kPi * length(uv), atan2f(uv.y, uv.x)));
        return true;
    }

    __device__ bool Device::FisheyeCamera::RectilinearViewportToCartesian(const ivec2& viewportPos, vec3& cart) const
    {
        cart = PolarToCartesian(vec2(kPi * (512 - viewportPos.y) / 512.0f, kTwoPi * viewportPos.x / 512.0f));
        return true;
    }

    __device__ bool Device::FisheyeCamera::LambertViewportToCartesian(const ivec2& viewportPos, vec3& cart) const
    {
        vec2 v = 2.0f * (vec2(viewportPos.x, 512 - viewportPos.y) - 256.0f) / 256.0f;
        if (length2(v) >= 4.0f) { return false; }

        const float a = sqrtf(1 - (v.x * v.x + v.y * v.y) / 4);
        cart = vec3(a * v.x, a * v.y, (v.x * v.x + v.y * v.y) * 0.5 - 1);
        return true;
    }

    __device__ void Device::FisheyeCamera::CreateRay(const ivec2& viewportPos, CompressedRay& ray) const
    {
        vec3 cartesian;
        if (!RectilinearViewportToCartesian(viewportPos, cartesian))
        {
            ray.Kill();
            return;
        }

        ray.SetViewportPos(viewportPos);
        ray.od.d = NormalToWorldSpace(cartesian, m_params.transform);
        ray.od.o = m_params.transform.trans;
        ray.sampleIdx++;
        ray.weight = kOne;
        ray.depth = 2;
        ray.flags = kRayLightProbe | kRayIndirectSample;
    }

    __device__ void Device::FisheyeCamera::Accumulate(RenderCtx& ctx, const vec3& value)
    {
        m_objects.cu_accumBuffer->Accumulate(ctx.emplacedRay.GetViewportPos(), value, ctx.emplacedRay.IsAlive());
    }

    __host__ Host::FisheyeCamera::FisheyeCamera(const ::Json::Node& parentNode, const std::string& id) :
        Host::Camera(parentNode, id)
    {
        // Create the accumulation buffer
        m_hostAccumBuffer = AssetHandle<Host::ImageRGBW>(tfm::format("%s_fisheyeAccumBuffer", id), 512, 512, m_hostStream);
        m_hostAccumBuffer->Clear(vec4(0.0f));

        // Instantiate the camera object on the device
        cu_deviceData = InstantiateOnDevice<Device::FisheyeCamera>();
        FromJson(parentNode, ::Json::kRequiredWarn);

        // Sychronise the device objects
        Device::FisheyeCamera::Objects deviceObjects;
        deviceObjects.cu_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        deviceObjects.renderState.cu_compressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
        deviceObjects.renderState.cu_blockRayOccupancy = m_hostBlockRayOccupancy->GetDeviceInstance();
        deviceObjects.renderState.cu_renderStats = m_hostRenderStats->GetDeviceInstance();
        SynchroniseObjects(cu_deviceData, deviceObjects);

        m_blockSize = dim3(16, 16, 1);
        m_gridSize = dim3((512 + 15) / 16, (512 + 15) / 16, 1);
    }

    __host__ AssetHandle<Host::RenderObject> Host::FisheyeCamera::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kCamera) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::FisheyeCamera(json, id), id);
    }

    __host__ void Host::FisheyeCamera::OnDestroyAsset()
    {
        Host::Camera::OnDestroyAsset();

        m_hostAccumBuffer.DestroyAsset();

        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::FisheyeCamera::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::RenderObject::UpdateDAGPath(parentNode);

        m_params = FisheyeCameraParams(parentNode);
        SynchroniseObjects(cu_deviceData, m_params);
    }

    __host__ void Host::FisheyeCamera::ClearRenderState()
    {
        m_hostAccumBuffer->Clear(vec4(0.0f));
        m_hostCompressedRayBuffer->Clear(Cuda::CompressedRay());
        //m_hostPixelFlagsBuffer->Clear(0);
    }

    __global__ void KernelSeedRayBuffer(Device::FisheyeCamera* camera)
    {
        camera->SeedRayBuffer(kKernelPos<ivec2>());
    }

    __host__ void Host::FisheyeCamera::OnPreRenderPass(const float wallTime, const float frameIdx)
    {
        KernelSeedRayBuffer << < m_gridSize, m_blockSize, 0, m_hostStream >> > (cu_deviceData);
    }

    __global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::FisheyeCamera* camera)
    {
        //if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

        camera->Composite(kKernelPos<ivec2>(), deviceOutputImage);
    }

    __host__ void Host::FisheyeCamera::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        hostOutputImage->SignalSetWrite(m_hostStream);
        KernelComposite << < m_blockSize, m_gridSize, 0, m_hostStream >> > (hostOutputImage->GetDeviceInstance(), cu_deviceData);
        hostOutputImage->SignalUnsetWrite(m_hostStream);
    }
}