#define CUDA_DEVICE_ASSERTS

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

        gridDensity = clamp(gridDensity, ivec3(2), ivec3(1000));
        shOrder = clamp(shOrder, 0, 2);
    }   

    __device__ Device::LightProbeCamera::LightProbeCamera() {  }

    __device__ void Device::LightProbeCamera::SeedRayBuffer()
    {
        CudaDeviceAssert(kKernelIdx < 512 * 512);
        
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

        const auto& texel = (*m_objects.cu_reduceBuffer)[viewportPos.y * 512 + viewportPos.x];

        // Normalise and gamma correct
        const vec3 rgb = texel.xyz / fmax(1.0f, texel.w);
        deviceOutputImage->At(viewportPos) = vec4(rgb, 1.0f);
    }

    __device__ void Device::LightProbeCamera::Prepare()
    {
        CudaDeviceAssert(m_objects.cu_accumBuffer);
        
        // Number of coefficients per probe
        m_coefficientsPerProbe = SH::GetNumCoefficients(m_params.shOrder);
        // Number of light probes in the grid
        m_numProbes = Volume(m_params.gridDensity);
        // Number of sample buckets per probe
        m_bucketsPerProbe = m_objects.cu_accumBuffer->Size() / m_numProbes;
        // Number of sample buckets per SH coefficient per probe
        m_bucketsPerCoefficient = m_bucketsPerProbe / m_coefficientsPerProbe;

        // Adjust values so everything packs correctly
        m_bucketsPerProbe = m_bucketsPerCoefficient * m_coefficientsPerProbe;
        m_totalBuckets = m_bucketsPerProbe * m_numProbes;

        // Used when parallel reducing the accumluation buffer
        for (m_bucketsPerCoefficientPow2 = 1; 
            m_bucketsPerCoefficientPow2 < m_bucketsPerCoefficient >> 1; 
            m_bucketsPerCoefficientPow2 <<= 1) {}

        CudaPrintVar(m_coefficientsPerProbe, i);
        CudaPrintVar(m_numProbes, i);
        CudaPrintVar(m_bucketsPerProbe, i);
        CudaPrintVar(m_bucketsPerCoefficient, i);
        CudaPrintVar(m_totalBuckets, i);
    }

    __device__ void Device::LightProbeCamera::CreateRay(const uint& accumIdx, CompressedRay& ray) const
    {           
        // Update the ray with the new properties and generate a random sampler from it
        ray.accumIdx = accumIdx;
        ray.sampleIdx++;
        ray.depth = 0;
        RNG rng(ray);

        ray.od.d = SampleUnitSphere(rng.Rand<0, 1>());

        // For i'th probe, coefficients are packed together:
        // [ [ C0(i), ..., C0(i) ] [ C1(i), ..., C1(i) ], ... , [ Cn(i), ..., Cn(i)] ]

        const int probeIdx = accumIdx / m_bucketsPerProbe;
        const int coeffIdx = (accumIdx / m_bucketsPerCoefficient) % m_coefficientsPerProbe;
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

    __device__ void Device::LightProbeCamera::ReduceProbeGrid()
    {
        CudaDeviceAssert(m_objects.cu_accumBuffer);
        CudaDeviceAssert(m_objects.cu_reduceBuffer);
        
        if (kKernelIdx >= m_totalBuckets) { return; }

        auto& accumBuffer = *m_objects.cu_accumBuffer;
        auto& reduceBuffer = *m_objects.cu_reduceBuffer;        
        const int subsampleIdx = kKernelIdx % m_bucketsPerCoefficient;

        for (uint i = m_bucketsPerCoefficientPow2; i > 0; i >>= 1)
        {
            if (subsampleIdx < i)
            {          
                if (i == m_bucketsPerCoefficientPow2)
                {
                    reduceBuffer[kKernelIdx] = accumBuffer[kKernelIdx];
                    if(subsampleIdx + i < m_bucketsPerCoefficient)
                    {
                        CudaDeviceAssert(kKernelIdx + i < 512 * 512);
                        reduceBuffer[kKernelIdx] += accumBuffer[kKernelIdx + i];
                    }
                }
                else
                {
                    CudaDeviceAssert(kKernelIdx + i < 512 * 512);
                    reduceBuffer[kKernelIdx] += reduceBuffer[kKernelIdx + i];
                }
            }
            __syncthreads();
        } 
         
        // Normalise the reduced values
        if (subsampleIdx == 0)
        {
            auto& texel = reduceBuffer[kKernelIdx];
            texel /= max(1.0f, texel.w);
        }
        else
        {
            reduceBuffer[kKernelIdx] = 0.0f;
        }        
    }

    __host__ Host::LightProbeCamera::LightProbeCamera(const ::Json::Node& parentNode, const std::string& id) :
        Host::Camera(parentNode, id)
    {
        // Create the accumulation and reduce buffers
        m_hostAccumBuffer = AssetHandle<Host::Array<vec4>>(tfm::format("%s_probeAccumBuffer", id), 512 * 512, m_hostStream);
        m_hostAccumBuffer->Clear(vec4(0.0f));
        m_hostReduceBuffer = AssetHandle<Host::Array<vec4>>(tfm::format("%s_probeReduceBuffer", id), 512 * 512, m_hostStream);

        //m_hostLightProbeGrid = AssetHandle<Host::LightProbeGrid>(tfm::format("%s_probeGrid", id), m_hostStream);

        // Instantiate the camera object on the device
        cu_deviceData = InstantiateOnDevice<Device::LightProbeCamera>();

        // Sychronise the device objects
        m_deviceObjects.cu_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.cu_reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
        m_deviceObjects.renderState.cu_compressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
        m_deviceObjects.renderState.cu_blockRayOccupancy = m_hostBlockRayOccupancy->GetDeviceInstance();
        m_deviceObjects.renderState.cu_renderStats = m_hostRenderStats->GetDeviceInstance();

        // Objects are re-synchronised at every JSON update
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
        m_hostReduceBuffer.DestroyAsset();

        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::LightProbeCamera::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::RenderObject::UpdateDAGPath(parentNode);
        
        m_params.FromJson(parentNode, flags);

        // Reduce the size of the grid if it exceeds the size of the accumulation buffer
        const int maxNumProbes = 512 * 512;
        if (Volume(m_params.gridDensity) > maxNumProbes)
        {
            const auto oldDensity = m_params.gridDensity;
            while (Volume(m_params.gridDensity) > maxNumProbes)
            {
                if (m_params.gridDensity.x > 1) { m_params.gridDensity.x--; }
                if (m_params.gridDensity.y > 1) { m_params.gridDensity.y--; }
                if (m_params.gridDensity.z > 1) { m_params.gridDensity.z--; }
            }
            Log::Error("WARNING: The size of the probe grid %s is too large for the accumulation buffer. Reducing to %s.\n", oldDensity.format(), m_params.gridDensity.format());
        }

        SynchroniseObjects(cu_deviceData, m_deviceObjects);
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

    __host__ void Host::LightProbeCamera::OnPreRenderPass(const float wallTime, const float frameIdx)
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

    __global__ void KernelReduceProbeGrid(Device::LightProbeCamera* camera)
    {
        camera->ReduceProbeGrid();
    }

    __host__ void Host::LightProbeCamera::OnPostRenderPass()
    {
        KernelReduceProbeGrid << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData);
    }
}