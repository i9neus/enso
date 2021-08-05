#include "CudaLightProbeCamera.cuh"
#include "generic/JsonUtils.h"

#include "../CudaCtx.cuh"
#include "../CudaManagedArray.cuh"
#include "../CudaManagedObject.cuh"

#include "../math/CudaSphericalHarmonics.cuh"

namespace Cuda
{
    __host__ __device__ LightProbeCameraParams::LightProbeCameraParams()
    {
        gridDensity = ivec3(5, 5, 5);
        shL = 1;
    }

    __host__ LightProbeCameraParams::LightProbeCameraParams(const ::Json::Node& node) :
        LightProbeCameraParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void LightProbeCameraParams::ToJson(::Json::Node& node) const
    {
        transform.ToJson(node);

        node.AddArray("gridDensity", std::vector<int>({ gridDensity.x, gridDensity.y, gridDensity.z }));
        node.AddValue("L", shL);
    }

    __host__ void LightProbeCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        transform.FromJson(node, flags);

        node.GetVector("gridDensity", gridDensity, flags);
        node.GetValue("L", shL, flags);
    }   

    __device__ Device::LightProbeCamera::LightProbeCamera()
    {
        Prepare();
    }

    __device__ void Device::LightProbeCamera::SeedRayBuffer()
    {
        assert(kKernelIdx < 512 * 512);

        if (kKernelIdx > m_totalBuckets) { return; }
        
        CompressedRay& compressedRay = (*m_objects.renderState.cu_compressedRayBuffer)[kKernelIdx];

        if (!compressedRay.IsAlive())
        {
            CreateRay(kKernelIdx, compressedRay);
        }
    }

    __device__ void Device::LightProbeCamera::Prepare()
    {
        // Number of light probes in the grid
        m_numProbes = m_params.gridDensity.x * m_params.gridDensity.y * m_params.gridDensity.z;
        // Number of sample buckets per probe
        m_bucketsPerProbe = m_objects.cu_accumBuffer->Size() / m_numProbes;
        // Number of sample buckets per SH coefficient per probe
        m_bucketsPerCoefficient = m_bucketsPerProbe / SH::GetNumCoefficients(m_params.shL);

        // Adjust values so everything packs correctly
        m_bucketsPerProbe = m_bucketsPerCoefficient * SH::GetNumCoefficients(m_params.shL);        
        m_totalBuckets = m_bucketsPerProbe * m_numProbes;
    }

    __device__ void Device::LightProbeCamera::CreateRay(const uint& accumIdx, CompressedRay& ray) const
    {           
        // Update the ray with the new properties and generate a random sampler from it
        ray.accumIdx = accumIdx;
        ray.sampleIdx++;
        ray.depth = 0;
        RNG rng(ray);

        ray.od.d = SampleUnitSphere(rng.Rand<0, 1>());      

        const int probeIdx = accumIdx / m_bucketsPerProbe;
        const int coeffIdx = accumIdx % m_bucketsPerCoefficient;
        const ivec3 gridIdx(probeIdx % m_params.gridDensity.x,
                            (probeIdx / m_params.gridDensity.x) % m_params.gridDensity.y,
                            probeIdx / (m_params.gridDensity.x * m_params.gridDensity.y));

        // Project this direction into SH and pre-normalise
        ray.weight = SH::Project(ray.od.d, coeffIdx) * kFourPi;
        ray.depth = 0;
        ray.flags = kRaySpecular;
        ray.od.o = m_params.transform.PointToWorldSpace(vec3(gridIdx) / vec3(m_params.gridDensity) - vec3(0.5f));
    }

    __device__ void Device::LightProbeCamera::Accumulate(RenderCtx& ctx, const vec3& value)
    {
        (*m_objects.cu_accumBuffer)[ctx.emplacedRay.accumIdx] += vec4(value, float(1 >> ctx.depth));
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
        camera->SeedRayBuffer();
    }

    __host__ void Host::LightProbeCamera::SeedRayBuffer()
    {
        KernelSeedRayBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData);
    }
}