#include "../CudaSampler.cuh"
#include "../CudaHash.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRay.cuh"

#include "CudaLightProbeCamera.cuh"
#include "generic/JsonUtils.h"
#include "../CudaManagedArray.cuh"
#include "../CudaManagedObject.cuh"

#define kCameraAA                 1.5f             // The width/height of the anti-aliasing kernel in pixels
#define kCameraSensorSize         0.035f           // The size of the camera sensor in meters
#define kBlades                   5.0f
#define kBladeCurvature           0.0f
#define kCameraUp                 vec3(0.0f, 1.0f, 0.0f) 

namespace Cuda
{
    __host__ __device__ LightProbeCameraParams::LightProbeCameraParams()
    {
        density = ivec3(5, 5, 5);      
    }

    __host__ LightProbeCameraParams::LightProbeCameraParams(const ::Json::Node& node) :
        LightProbeCameraParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void LightProbeCameraParams::ToJson(::Json::Node& node) const
    {
        transform.ToJson(node);
        
        node.AddArray("density", std::vector<int>({ density.x, density.y, density.z }));     
    }

    __host__ void LightProbeCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        transform.FromJson(node, flags);
        
        node.GetVector("density", density, flags);
    }   

    __device__ Device::LightProbeCamera::LightProbeCamera()
    {
        Prepare();
    }

    __device__ void Device::LightProbeCamera::SeedRayBuffer(const ivec2& viewportPos)
    {
        CompressedRay& compressedRay = (*m_objects.renderState.cu_compressedRayBuffer)[viewportPos.y * 512 + viewportPos.x];

        if (!compressedRay.IsAlive())
        {
            CreateRay(viewportPos, compressedRay);
        }
    }

    __device__ void Device::LightProbeCamera::Prepare()
    {
        
    }

    __device__ void Device::LightProbeCamera::CreateRay(const ivec2& viewportPos, CompressedRay& ray) const
    {
       
    }

    __device__ void Device::LightProbeCamera::Accumulate(RenderCtx& ctx, const vec3& value)
    {
        
    }

    __host__ Host::LightProbeCamera::LightProbeCamera(const ::Json::Node& parentNode, const std::string& id) :
        Host::Camera(parentNode, id)
    {
        // Create the accumulation buffer
        m_hostAccumBuffer = AssetHandle<Host::Array<vec4>>(tfm::format("%s_probeAccumBuffer", id), 512 * 512, m_hostStream);
        m_hostAccumBuffer->Clear(vec4(0.0f));

        // Instantiate the camera object on the device
        cu_deviceData = InstantiateOnDevice<Device::LightProbeCamera>();
        FromJson(parentNode, ::Json::kRequiredWarn);

        // Sychronise the device objects
        Device::LightProbeCamera::Objects objects;
        objects.cu_accumBuffer = nullptr;
        objects.renderState.cu_compressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
        objects.renderState.cu_blockRayOccupancy = m_hostBlockRayOccupancy->GetDeviceInstance();
        objects.renderState.cu_renderStats = m_hostRenderStats->GetDeviceInstance();
        SynchroniseObjects(cu_deviceData, objects);

        m_block = dim3(16 * 16, 1, 1);
        m_grid = dim3(512 * 512 / m_block.x , 1, 1);
    }

    __host__ AssetHandle<Host::RenderObject> Host::LightProbeCamera::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kCamera) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LightProbeCamera(json, id), id);
    }

    __host__ void Host::LightProbeCamera::OnDestroyAsset()
    {
        Host::Camera::OnDestroyAsset();

        m_hostAccumBuffer.DestroyAsset();

        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::LightProbeCamera::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::Camera::FromJson(parentNode, flags);

        SynchroniseObjects(cu_deviceData, LightProbeCameraParams(parentNode));
    }

    __host__ void Host::LightProbeCamera::ClearRenderState()
    {
        m_hostAccumBuffer->Clear(vec4(0.0f));
        m_hostCompressedRayBuffer->Clear(Cuda::CompressedRay());
        //m_hostPixelFlagsBuffer->Clear(0);
    }

    __global__ void KernelSeedRayBuffer(Device::LightProbeCamera* camera)
    {
        camera->SeedRayBuffer(kKernelPos<ivec2>());
    }

    __host__ void Host::LightProbeCamera::SeedRayBuffer()
    {
        KernelSeedRayBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData);
    }
}