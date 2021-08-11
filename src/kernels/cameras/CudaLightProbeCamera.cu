#define CUDA_DEVICE_ASSERTS

#include "CudaLightProbeCamera.cuh"
#include "generic/JsonUtils.h"

#include "../CudaCtx.cuh"
#include "../CudaManagedArray.cuh"
#include "../CudaManagedObject.cuh"

#include "../math/CudaSphericalHarmonics.cuh"

namespace Cuda
{
    __host__ __device__ LightProbeCameraParams::LightProbeCameraParams() {}

    __host__ LightProbeCameraParams::LightProbeCameraParams(const ::Json::Node& node) :
        LightProbeCameraParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void LightProbeCameraParams::ToJson(::Json::Node& node) const
    {
        grid.ToJson(node);
        camera.ToJson(node);
    }

    __host__ void LightProbeCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        grid.FromJson(node, flags);
        camera.FromJson(node, flags);
    }   

    __device__ Device::LightProbeCamera::LightProbeCamera() {  }

    __device__ void Device::LightProbeCamera::Synchronise(const LightProbeCameraParams& params)
    {
        m_params = params;
        Prepare();
    }
    __device__ void Device::LightProbeCamera::Synchronise(const Objects& objects)
    {
        m_objects = objects;
    }

    __device__ void Device::LightProbeCamera::SeedRayBuffer()
    {
        CudaDeviceAssert(kKernelIdx < 512 * 512);
        
        CompressedRay& compressedRay = (*m_objects.renderState.cu_compressedRayBuffer)[kKernelIdx];

        if (kKernelIdx > m_params.totalBuckets) 
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
        const float lum = mean(rgb);        
        deviceOutputImage->At(viewportPos) = vec4(((lum < 0.0f) ? kRed : kGreen) * lum, 1.0f);
    }

    __device__ void Device::LightProbeCamera::Prepare()
    {
        CudaDeviceAssert(m_objects.cu_accumBuffer);
    }

    __device__ void Device::LightProbeCamera::GetProbeAttributesFromIndex(const uint& accumIdx, int& probeIdx, int& coeffIdx, ivec3& gridIdx) const
    {
        probeIdx = accumIdx / m_params.bucketsPerProbe;
        coeffIdx = (accumIdx / m_params.bucketsPerCoefficient) % m_params.coefficientsPerProbe;
        gridIdx = ivec3(probeIdx % m_params.grid.gridDensity.x,
            (probeIdx / m_params.grid.gridDensity.x) % m_params.grid.gridDensity.y,
            probeIdx / (m_params.grid.gridDensity.x * m_params.grid.gridDensity.y));
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

        int probeIdx, coeffIdx;
        ivec3 gridIdx;
        GetProbeAttributesFromIndex(accumIdx, probeIdx, coeffIdx, gridIdx);

        // Project this direction into SH and pre-normalise
        ray.weight = SH::Project(ray.od.d, coeffIdx) * kFourPi;
        ray.depth = 0;
        ray.flags = kRayLightProbe | kRayIndirectSample;
        ray.od.o = m_params.grid.transform.PointToWorldSpace(vec3(gridIdx) / vec3(m_params.grid.gridDensity) - vec3(0.5f));
    }

    __device__ void Device::LightProbeCamera::Accumulate(RenderCtx& ctx, const vec3& value)
    {
        if (m_params.grid.debugBakePRef)
        {
            int probeIdx, coeffIdx;
            ivec3 gridIdx;
            GetProbeAttributesFromIndex(ctx.emplacedRay.accumIdx, probeIdx, coeffIdx, gridIdx);

            const vec3 p = vec3(gridIdx) / vec3(m_params.grid.gridDensity - ivec3(1));
            (*m_objects.cu_accumBuffer)[ctx.emplacedRay.accumIdx] += vec4(p, float(1 >> ctx.depth));
        }
        else
        {
            (*m_objects.cu_accumBuffer)[ctx.emplacedRay.accumIdx] += vec4(value, float(1 >> ctx.depth));
        }    
    }

    __device__ void Device::LightProbeCamera::ReduceAccumulationBuffer(const uint batchSize, const uvec2 batchRange)
    {
        CudaDeviceAssert(m_objects.cu_accumBuffer);
        CudaDeviceAssert(m_objects.cu_reduceBuffer);
        CudaDeviceAssert(m_objects.cu_probeGrid);

        if (batchRange[0] == batchSize) (*m_objects.cu_reduceBuffer)[kKernelIdx] = 0.0f;

        if (kKernelIdx >= m_params.totalBuckets) { return; }

        auto& accumBuffer = *m_objects.cu_accumBuffer;
        auto& reduceBuffer = *m_objects.cu_reduceBuffer;
        const int subsampleIdx = kKernelIdx % m_params.bucketsPerCoefficient;

        for (uint iterationSize = batchRange[0] / 2; iterationSize > batchRange[1] / 2; iterationSize >>= 1)
        {
            if (subsampleIdx < iterationSize)
            {
                // For the first iteration, copy the data out of the accumulation buffer
                if (iterationSize == batchSize / 2)
                {
                    auto& texel = reduceBuffer[kKernelIdx];
                    texel = accumBuffer[kKernelIdx];
                    if (subsampleIdx + iterationSize < m_params.bucketsPerCoefficient)
                    {
                        CudaDeviceAssert(kKernelIdx + iterationSize < 512 * 512);
                        texel += accumBuffer[kKernelIdx + iterationSize];
                    }
                    texel /= max(1.0f, texel.w);
                }
                else
                {
                    CudaDeviceAssert(kKernelIdx + iterationSize < 512 * 512);
                    CudaDeviceAssert(subsampleIdx + iterationSize < m_params.bucketsPerCoefficient);
                    reduceBuffer[kKernelIdx] += reduceBuffer[kKernelIdx + iterationSize];
                }
            }
            else
            {
                //reduceBuffer[kKernelIdx] = 1.0f;
            }

            __syncthreads();
            break;
        } 

        // After the last operation, cache the accumulated value in the probe grid
        if (subsampleIdx == 0 && batchRange[1] == 2)
        {
            auto& texel = reduceBuffer[kKernelIdx];
            texel /= max(1.0f, texel.w);

            const int probeIdx = kKernelIdx / m_params.bucketsPerProbe;
            const int coeffIdx = (kKernelIdx / m_params.bucketsPerCoefficient) % m_params.coefficientsPerProbe;
            m_objects.cu_probeGrid->SetSHCoefficient(probeIdx, coeffIdx, texel.xyz);
        }
    }

    __host__ Host::LightProbeCamera::LightProbeCamera(const ::Json::Node& parentNode, const std::string& id) :
        Host::Camera(parentNode, id)
    {
        // Create the accumulation and reduce buffers
        m_hostAccumBuffer = AssetHandle<Host::Array<vec4>>(tfm::format("%s_probeAccumBuffer", id), 512 * 512, m_hostStream);
        m_hostAccumBuffer->Clear(vec4(0.0f));
        m_hostReduceBuffer = AssetHandle<Host::Array<vec4>>(tfm::format("%s_probeReduceBuffer", id), 512 * 512, m_hostStream);

        const std::string gridId = tfm::format("%s_probeGrid", id);
        m_hostLightProbeGrid = AssetHandle<Host::LightProbeGrid>(gridId, gridId);

        // Instantiate the camera object on the device
        cu_deviceData = InstantiateOnDevice<Device::LightProbeCamera>();

        // Sychronise the device objects
        m_deviceObjects.cu_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        m_deviceObjects.cu_reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
        m_deviceObjects.cu_probeGrid = m_hostLightProbeGrid->GetDeviceInstance();
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

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::LightProbeCamera::GetChildObjectHandles() 
    { 
        std::vector<AssetHandle<Host::RenderObject>> children;
        if (m_hostLightProbeGrid)
        {
            children.push_back(AssetHandle<Host::RenderObject>(m_hostLightProbeGrid));
        }
        return children;
    }

    __host__ void Host::LightProbeCamera::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::RenderObject::UpdateDAGPath(parentNode);
        
        m_params.FromJson(parentNode, flags);

        // Reduce the size of the grid if it exceeds the size of the accumulation buffer
        const int maxNumProbes = 512 * 512;
        if (Volume(m_params.grid.gridDensity) > maxNumProbes)
        {
            const auto oldDensity = m_params.grid.gridDensity;
            while (Volume(m_params.grid.gridDensity) > maxNumProbes)
            {
                m_params.grid.gridDensity = max(ivec3(1), m_params.grid.gridDensity - ivec3(1));
            }
            Log::Error("WARNING: The size of the probe grid %s is too large for the accumulation buffer. Reducing to %s.\n", oldDensity.format(), m_params.grid.gridDensity.format());
        }

        // Prepare the light probe grid with the new parameters
        m_hostLightProbeGrid->Prepare(m_params.grid);

        // Number of coefficients per probe
        m_params.coefficientsPerProbe = SH::GetNumCoefficients(m_params.grid.shOrder);
        // Number of light probes in the grid
        m_params.numProbes = Volume(m_params.grid.gridDensity);
        // Number of sample buckets per probe
        m_params.bucketsPerProbe = m_hostAccumBuffer->Size() / m_params.numProbes;
        // Number of sample buckets per SH coefficient per probe
        m_params.bucketsPerCoefficient = m_params.bucketsPerProbe / m_params.coefficientsPerProbe;

        // Adjust values so everything packs correctly
        m_params.bucketsPerProbe = m_params.bucketsPerCoefficient * m_params.coefficientsPerProbe;
        m_params.totalBuckets = m_params.bucketsPerProbe * m_params.numProbes;   

        // Used when parallel reducing the accumluation buffer
        uint reduceBatchSizePow2 = 1;
        for (; reduceBatchSizePow2 < m_params.bucketsPerCoefficient;
            reduceBatchSizePow2 <<= 1) {
        }

        Log::Debug("coefficientsPerProbe: %i\n", m_params.coefficientsPerProbe);
        Log::Debug("numProbes: %i\n", m_params.numProbes);
        Log::Debug("bucketsPerProbe: %i\n", m_params.bucketsPerProbe);
        Log::Debug("bucketsPerCoefficient: %i\n", m_params.bucketsPerCoefficient);
        Log::Debug("bucketsPerProbe: %i\n", m_params.bucketsPerProbe);
        Log::Debug("totalBuckets: %i\n", m_params.totalBuckets);
        Log::Debug("reduceBatchSizePow2: %i\n", reduceBatchSizePow2);

        // Sync everything with the device
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

    __global__ void KernelReduceAccumulationBuffer(Device::LightProbeCamera* camera, const uint reduceBatchSize, const uvec2 batchRange)
    {
        camera->ReduceAccumulationBuffer(reduceBatchSize, batchRange);
    }

    __host__ void Host::LightProbeCamera::OnPostRenderPass()
    {
        // Used when parallel reducing the accumluation buffer
        uint reduceBatchSizePow2 = 1;
        for (;reduceBatchSizePow2 < m_params.bucketsPerCoefficient;
            reduceBatchSizePow2 <<= 1) {
        }
        
        // Reduce until the batch range is equal to the size of the block
        uint batchSize = reduceBatchSizePow2;
        while (batchSize > 16 * 16)
        {
            KernelReduceAccumulationBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData, reduceBatchSizePow2, uvec2(batchSize, batchSize >> 1));
            batchSize >>= 1;
        }
        // Reduce the block in a single operation
        KernelReduceAccumulationBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData, reduceBatchSizePow2, uvec2(batchSize, 2));

        IsOk(cudaStreamSynchronize(m_hostStream));
    }
}