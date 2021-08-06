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
        shOrder = 1;
    }

    __host__ LightProbeCameraParams::LightProbeCameraParams(const ::Json::Node& node) :
        LightProbeCameraParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void LightProbeCameraParams::ToJson(::Json::Node& node) const
    {
        transform.ToJson(node);
        camera.ToJson(node);

        node.AddArray("gridDensity", std::vector<int>({ gridDensity.x, gridDensity.y, gridDensity.z }));
        node.AddValue("shOrder", shOrder);
    }

    __host__ void LightProbeCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        transform.FromJson(node, flags);
        camera.FromJson(node, flags);

        node.GetVector("gridDensity", gridDensity, flags);
        node.GetValue("shOrder", shOrder, flags);
    }   

    __device__ Device::LightProbeCamera::LightProbeCamera()
    {

    }

    __device__ void Device::LightProbeCamera::SeedRayBuffer()
    {
        assert(kKernelIdx < 512 * 512);
        
        CompressedRay& compressedRay = (*m_objects.renderState.cu_compressedRayBuffer)[kKernelIdx];

        if (kKernelIdx > m_totalBuckets) 
        {
            compressedRay.Kill();
            return; 
        }

        if (!compressedRay.IsAlive())
        {
            CreateRay(kKernelIdx, compressedRay);
        }
    }

    __device__ void Device::LightProbeCamera::Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const
    {
        if (!m_objects.cu_accumBuffer || viewportPos.x >= deviceOutputImage->Width() || viewportPos.y >= deviceOutputImage->Height() ||
            viewportPos.x >= 512 || viewportPos.y >= 512) {
            return;
        }

        const auto& texel = (*m_objects.cu_accumBuffer)[viewportPos.y * 512 + viewportPos.x];

        // Normalise and gamma correct
        const vec3 rgb = texel.xyz / fmax(1.0f, texel.w);
        deviceOutputImage->At(viewportPos) = vec4(rgb, 1.0f);
    }

    __device__ void Device::LightProbeCamera::Prepare()
    {
        if (!m_objects.cu_accumBuffer) return;
        
        // Number of coefficients per probe
        m_coefficientsPerProbe = SH::GetNumCoefficients(m_params.shOrder);
        // Number of light probes in the grid
        m_numProbes = m_params.gridDensity.x * m_params.gridDensity.y * m_params.gridDensity.z;
        // Number of sample buckets per probe
        m_bucketsPerProbe = m_objects.cu_accumBuffer->Size() / m_numProbes;
        // Number of sample buckets per SH coefficient per probe
        m_bucketsPerCoefficient = m_bucketsPerProbe / m_coefficientsPerProbe;

        // Adjust values so everything packs correctly
        m_bucketsPerProbe = m_bucketsPerCoefficient * m_coefficientsPerProbe;
        m_totalBuckets = m_bucketsPerProbe * m_numProbes;

        printf("m_numProbes: %u\n", m_numProbes);
        printf("m_bucketsPerProbe: %u\n", m_bucketsPerProbe);
        printf("m_bucketsPerCoefficient: %u\n", m_bucketsPerCoefficient);
        printf("m_totalBuckets: %u\n", m_totalBuckets);
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
        const int coeffIdx = accumIdx % m_coefficientsPerProbe;
        const ivec3 gridIdx(probeIdx % m_params.gridDensity.x,
                            (probeIdx / m_params.gridDensity.x) % m_params.gridDensity.y,
                            probeIdx / (m_params.gridDensity.x * m_params.gridDensity.y));

        // Project this direction into SH and pre-normalise
        ray.weight = SH::Project(ray.od.d, coeffIdx) * kFourPi;
        ray.depth = 0;
        ray.flags = kRayLightProbe | kRayIndirectSample;
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

        // Sychronise the device objects
        m_deviceObjects.cu_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.renderState.cu_compressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
        m_deviceObjects.renderState.cu_blockRayOccupancy = m_hostBlockRayOccupancy->GetDeviceInstance();
        m_deviceObjects.renderState.cu_renderStats = m_hostRenderStats->GetDeviceInstance();
        SynchroniseObjects(cu_deviceData, m_deviceObjects);

        FromJson(parentNode, ::Json::kRequiredWarn);

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
        Host::RenderObject::UpdateDAGPath(parentNode);
        
        m_params.FromJson(parentNode, flags);
        SynchroniseObjects(cu_deviceData, m_params);
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

    __global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::LightProbeCamera* camera)
    {
        //if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

        camera->Composite(kKernelPos<ivec2>(), deviceOutputImage);
    }

    __host__ void Host::LightProbeCamera::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize = dim3(16, 16, 1);
        dim3 gridSize(512 / 16, 512 / 16, 1);
        
        hostOutputImage->SignalSetWrite(m_hostStream);
        KernelComposite << < blockSize, gridSize, 0, m_hostStream >> > (hostOutputImage->GetDeviceInstance(), cu_deviceData);
        hostOutputImage->SignalUnsetWrite(m_hostStream);
    }
}