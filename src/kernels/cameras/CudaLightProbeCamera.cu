#define CUDA_DEVICE_ASSERTS

#include "CudaLightProbeCamera.cuh"
#include "generic/JsonUtils.h"

#include "../CudaCtx.cuh"
#include "../CudaManagedArray.cuh"
#include "../CudaManagedObject.cuh"

#include "../math/CudaSphericalHarmonics.cuh"

#include "../../io/USDIO.h"

#define kAccumBufferWidth 1024u
#define kAccumBufferHeight 1024u
#define kAccumBufferSize (kAccumBufferWidth * kAccumBufferHeight)

#define kRayBufferWidth 512u
#define kRayBufferHeight 512u
#define kRayBufferSize (kRayBufferWidth * kRayBufferHeight)


namespace Cuda
{
    __host__ __device__ LightProbeCameraParams::LightProbeCameraParams()
    {

    }

    __host__ LightProbeCameraParams::LightProbeCameraParams(const ::Json::Node& node) :
        LightProbeCameraParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void LightProbeCameraParams::ToJson(::Json::Node& node) const
    {
        auto gridNode = node.AddChildObject("grid");
        grid.ToJson(gridNode);

        camera.ToJson(node);
    }

    __host__ void LightProbeCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        auto gridNode = node.GetChildObject("grid", flags);
        grid.FromJson(gridNode, flags);

        camera.FromJson(node, flags);
    }      

    __device__ Device::LightProbeCamera::LightProbeCamera() {  }

    __device__ void Device::LightProbeCamera::Synchronise(const LightProbeCameraParams& params)
    {
        m_params = params;
        if (m_params.camera.maxSamples == 0) { m_params.camera.maxSamples = INT_MAX; }

        Prepare();
    }
    __device__ void Device::LightProbeCamera::Synchronise(const Objects& objects)
    {
        m_objects = objects;
    }

    __device__ void Device::LightProbeCamera::SeedRayBuffer(const int frameIdx)
    {
        assert(kKernelIdx < kRayBufferSize);
        
        CompressedRay& compressedRay = (*m_objects.renderState.cu_compressedRayBuffer)[kKernelIdx];

        if (kKernelIdx > m_params.totalBuckets) 
        {
            compressedRay.Kill();
            return; 
        }

        // On the first frame, reset the ray and the sample index
        if (frameIdx == 0)
        {
            compressedRay.Reset();
            compressedRay.sampleIdx = m_seedOffset;
        }

        if (!compressedRay.IsAlive() && 
            (m_params.camera.maxSamples <= 0 || int(compressedRay.sampleIdx - m_seedOffset) < m_params.maxSamplesPerBucket))
        {
            //(*m_objects.cu_accumBuffer)[kKernelIdx] = 0.0f;
            CreateRay(kKernelIdx, compressedRay, frameIdx);
        }
    }

    __device__ void Device::LightProbeCamera::Composite(const ivec2& accumPos, Device::ImageRGBA* deviceOutputImage) const
    {
        const ivec2 viewportPos = accumPos + deviceOutputImage->Dimensions() / 2 - ivec2(kAccumBufferWidth, kAccumBufferHeight) / 2;
        if (viewportPos.x < 0 || viewportPos.x >= deviceOutputImage->Width() ||
            viewportPos.y < 0 || viewportPos.y >= deviceOutputImage->Height()) {
            return;
        }

        assert(accumPos.y * kAccumBufferWidth + accumPos.x <= m_objects.cu_reduceBuffer->Size());
        //assert(m_objects.cu_reduceBuffer->Size() == kAccumBufferSize);

        // Normalise and gamma correct
        const auto& texel = (*m_objects.cu_reduceBuffer)[accumPos.y * kAccumBufferWidth + accumPos.x];
        const vec3 rgb = texel.xyz / fmax(1.0f, texel.w);       
        deviceOutputImage->At(viewportPos) = vec4(rgb, 1.0f);
    }

    __device__ void Device::LightProbeCamera::Prepare()
    {
        assert(m_objects.cu_accumBuffer);

        // Only use the lower 31 bits for the seed because we need to deduce the actual sample count from it
        m_seedOffset = HashOf(m_params.camera.seed & ((1 << 31) - 1));
    }

    __device__ void Device::LightProbeCamera::CreateRay(const uint& subsampleIdx, CompressedRay& ray, const int frameIdx) const
    {
        assert(subsampleIdx < kAccumBufferSize);
        
        const int probeIdx = subsampleIdx / m_params.subsamplesPerProbe;
        const ivec3 gridIdx = GridIdxFromProbeIdx(probeIdx, m_params.grid.gridDensity);
        
        ray.accumIdx = subsampleIdx;
        ray.sampleIdx++;
        ray.depth = 1;
        RNG rng(ray);

        const vec2 xi = rng.Rand<0, 1>();
        ray.od.d = SampleUnitSphere(xi);
        ray.probeDir = ray.od.d;

        // Probes data is grouped as follows:
        // Subsample    = [SH data] [Auxilliary data]
        // Probe        = [Subsample 1] .... [Subsample N]
        // Grid         = [Probe 1] ... [Probe M]        

        // Project this direction into SH and pre-normalise
        ray.weight = kOne;
        ray.depth = 2;
        ray.flags = kRayLightProbe | kRayIndirectSample;

        ray.od.o = m_params.grid.aspectRatio * vec3(gridIdx) / vec3(m_params.grid.gridDensity - 1) - vec3(0.5f);
        ray.od.o = m_params.grid.transform.PointToWorldSpace(ray.od.o);
    }

    __device__ void Device::LightProbeCamera::Accumulate(RenderCtx& ctx, const HitCtx& hitCtx, const vec3& value)
    {        
        int accumIdx = ctx.emplacedRay.accumIdx * m_params.coefficientsPerProbe;
        assert(ctx.emplacedRay.accumIdx < kAccumBufferSize);

        auto& accumBuffer = *m_objects.cu_accumBuffer;
        const float weight = float(!ctx.emplacedRay.IsAlive());
        
        // Accumulate the SH coefficients
        for (int shIdx = 0; shIdx < m_params.coefficientsPerProbe - 1; ++shIdx, ++accumIdx)
        {
            accumBuffer[accumIdx] += vec4(value * SH::Project(ctx.emplacedRay.probeDir, shIdx) * kFourPi, weight);
        } 

        if (ctx.depth == 2)
        {
            // A probe sample is valid if, on the first hit, it intersects with a front-facing surface or it leaves the scene
            accumBuffer[accumIdx] += vec4(float(!hitCtx.isValid || !hitCtx.backfacing), 0.0f, 0.0f, 1.0f);
        }  
    }
    __device__ void Device::LightProbeCamera::ReduceAccumulatedSample(vec4& dest, const vec4& source)
    {              
        if (int(dest.w) >= m_params.maxSamplesPerProbe - 1) 
        {             
            return;
        }
        
        if (int(dest.w + source.w) < m_params.maxSamplesPerProbe)
        {           
            dest += source;
            return;
        }

        dest += source * (m_params.maxSamplesPerProbe - dest.w) / source.w;
    }

    __device__ void Device::LightProbeCamera::ReduceAccumulationBuffer(const uint batchSize, const uvec2 batchRange)
    {         
        if (kKernelIdx >= m_params.totalBuckets) { return; }

        assert(m_objects.cu_accumBuffer);
        assert(m_objects.cu_reduceBuffer);
        assert(m_objects.cu_probeGrid);

        //if (batchRange[0] == batchSize) (*m_objects.cu_reduceBuffer)[kKernelIdx] = 0.0f;

        auto& accumBuffer = *m_objects.cu_accumBuffer;
        auto& reduceBuffer = *m_objects.cu_reduceBuffer;
        
        const int probeIdx = kKernelIdx / m_params.bucketsPerProbe;
        const int probeSubsampleIdx = (kKernelIdx / m_params.coefficientsPerProbe) % m_params.subsamplesPerProbe;
        const int coeffIdx = kKernelIdx % m_params.coefficientsPerProbe;

        for (uint iterationSize = batchRange[0] / 2; iterationSize > batchRange[1] / 2; iterationSize >>= 1)
        {
            if (probeSubsampleIdx < iterationSize)
            {
                // For the first iteration, copy the data out of the accumulation buffer
                if (iterationSize == batchSize / 2)
                {
                    auto& texel = reduceBuffer[kKernelIdx];
                    texel = 0.0f;
                    ReduceAccumulatedSample(texel, accumBuffer[kKernelIdx]);

                    if (probeSubsampleIdx + iterationSize < m_params.subsamplesPerProbe)
                    {
                        assert(kKernelIdx + iterationSize * m_params.coefficientsPerProbe < kAccumBufferSize);
                        //if (probeIdx == 0 && coeffIdx == 0) { printf("%i: %f + %f = %f\n", iterationSize, texel.w, accumBuffer[kKernelIdx + iterationSize].w, texel.w + accumBuffer[kKernelIdx + iterationSize].w); }
                        ReduceAccumulatedSample(texel, accumBuffer[kKernelIdx + iterationSize * m_params.coefficientsPerProbe]);

                    }
                    //else
                    //    if (probeIdx == 0 && coeffIdx == 0) { printf("%i: %f\n", iterationSize, texel.w); }
                }
                else
                {
                    assert(kKernelIdx + iterationSize * m_params.coefficientsPerProbe < kAccumBufferSize);
                    assert(probeSubsampleIdx + iterationSize < m_params.subsamplesPerProbe);
                    //if (probeIdx == 0 && coeffIdx == 0) { printf("%i: %f + %f = %f\n", iterationSize, reduceBuffer[kKernelIdx].w, reduceBuffer[kKernelIdx + iterationSize].w, reduceBuffer[kKernelIdx].w + reduceBuffer[kKernelIdx + iterationSize].w); }
                    ReduceAccumulatedSample(reduceBuffer[kKernelIdx], reduceBuffer[kKernelIdx + iterationSize * m_params.coefficientsPerProbe]);
                }
            }
            else
            {
                //reduceBuffer[kKernelIdx] = 1.0f;
            }

            __syncthreads();
        } 

        // After the last operation, cache the accumulated value in the probe grid
        if (probeSubsampleIdx == 0 && batchRange[0] == 2)
        {
            auto& texel = reduceBuffer[kKernelIdx];           

            //const int probeIdx = kKernelIdx / m_params.bucketsPerProbe;
            //const int coeffIdx = (kKernelIdx / m_params.bucketsPerCoefficient) % m_params.coefficientsPerProbe;
            if (coeffIdx == m_params.coefficientsPerProbe - 1)
            {
               // Store the validity value and the sample count in the final coefficient
               texel.x /= max(1.0f, texel.w);
               texel.y = texel.w;
            }
            else
            {
                texel /= max(1.0f, texel.w);
            }

            m_objects.cu_probeGrid->SetSHCoefficient(probeIdx, coeffIdx, texel.xyz);
        }
    }

    __device__ vec2 Device::LightProbeCamera::GetProbeMinMaxSampleCount() const
    {
        __shared__ vec2 localMinMax[256];

        if (kKernelIdx == 0)
        {
            for (int i = 0; i < 256; i++) { localMinMax[i] = vec2(kFltMax, 0.0f); }
        }

        __syncthreads();

        const int startIdx = (m_params.numProbes - 1) * kKernelIdx / 256;
        const int endIdx = (m_params.numProbes - 1) * (kKernelIdx + 1) / 256;
        for (int i = startIdx; i <= endIdx; i++)
        {
            const float coeff = m_objects.cu_probeGrid->GetSHCoefficient(startIdx, m_params.coefficientsPerProbe - 1).y;
            localMinMax[kKernelIdx] = vec2(min(localMinMax[kKernelIdx].x, coeff), max(localMinMax[kKernelIdx].y, coeff));
        }        

        __syncthreads();

        vec2 globalMinMax = localMinMax[0];
        if (kKernelIdx == 0)
        {
            for (int i = 1; i < 256; i++) 
            { 
                globalMinMax = vec2(min(localMinMax[kKernelIdx].x, globalMinMax.x), max(localMinMax[kKernelIdx].y, globalMinMax.y));
            }
        }

        return globalMinMax;
    }

    __host__ Host::LightProbeCamera::LightProbeCamera(const ::Json::Node& parentNode, const std::string& id) :
        Host::Camera(parentNode, id),
        m_block(16 * 16, 1, 1),
        m_seedGrid(1, 1, 1),
        m_reduceGrid(1, 1, 1),
        m_exporterState(kDisarmed)
    {
        // Create the accumulation and reduce buffers
        m_hostAccumBuffer = AssetHandle<Host::Array<vec4>>(tfm::format("%s_probeAccumBuffer", id), kAccumBufferSize, m_hostStream);
        m_hostAccumBuffer->Clear(vec4(0.0f));
        m_hostReduceBuffer = AssetHandle<Host::Array<vec4>>(tfm::format("%s_probeReduceBuffer", id), kAccumBufferSize, m_hostStream);

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

    template<typename T>
    __host__ T NearestPow2Ceil(const T& j)
    {
        T i = 1;
        for (; i < j; i <<= 1) {};
        return i;
    }

    template<typename T>
    __host__ T NearestPow2Floor(const T& j)
    {
        T i = 1;
        for (; i <= j; i <<= 1) {};
        return i >> 1;
    }

    __host__ void Host::LightProbeCamera::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::RenderObject::UpdateDAGPath(parentNode);
        
        m_params.FromJson(parentNode, flags);

        Prepare();        
    }

    __host__ void Host::LightProbeCamera::Prepare()
    {
        m_params.coefficientsPerProbe = SH::GetNumCoefficients(m_params.grid.shOrder) + 1;
        
        // Reduce the size of the grid if it exceeds the size of the accumulation buffer
        const int maxNumProbes = min(kAccumBufferSize / m_params.coefficientsPerProbe, kRayBufferSize);
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

        
        // Number of light probes in the grid
        m_params.numProbes = Volume(m_params.grid.gridDensity);
        // Number of SH parameter sets per probe, reduced later to get the final value 
        m_params.subsamplesPerProbe = min(kRayBufferSize / m_params.numProbes, 
                                          kAccumBufferSize / (m_params.numProbes * m_params.coefficientsPerProbe));
        
        // The maximum number of samples per bucket based on the number of buckets per coefficient
        m_params.maxSamplesPerProbe = m_params.maxSamplesPerBucket = std::numeric_limits<int>::max();
        if (m_params.camera.maxSamples > 0)
        {
            m_params.maxSamplesPerProbe = m_params.camera.maxSamples;
            m_params.maxSamplesPerBucket = int(1.0f + float(m_params.camera.maxSamples) / float(m_params.subsamplesPerProbe));
        }

        // Derive some more values
        m_params.bucketsPerProbe = m_params.subsamplesPerProbe * m_params.coefficientsPerProbe;
        m_params.totalBuckets = m_params.bucketsPerProbe * m_params.numProbes;
        m_params.totalSubsamples = m_params.subsamplesPerProbe * m_params.numProbes;

        // Used when parallel reducing the accumluation buffer
        uint reduceBatchSizePow2 = NearestPow2Ceil(m_params.subsamplesPerProbe);

        Log::Debug("coefficientsPerProbe: %i\n", m_params.coefficientsPerProbe);
        Log::Debug("numProbes: %i\n", m_params.numProbes);
        Log::Debug("subsamplesPerProbe: %i\n", m_params.subsamplesPerProbe);
        Log::Debug("bucketsPerProbe: %i\n", m_params.bucketsPerProbe);
        Log::Debug("totalBuckets: %i\n", m_params.totalBuckets);
        Log::Debug("maxSamplesPerBucket: %i\n", m_params.maxSamplesPerBucket);
        Log::Debug("reduceBatchSizePow2: %i\n", reduceBatchSizePow2);

        // Sync everything with the device
        SynchroniseObjects(cu_deviceData, m_deviceObjects);
        SynchroniseObjects(cu_deviceData, m_params);

        const int seedGridSize = m_params.totalSubsamples;
        m_seedGrid = dim3((seedGridSize + (m_block.x - 1)) / m_block.x, 1, 1);
        const int reduceGridSize = m_params.totalBuckets;
        m_reduceGrid = dim3((reduceGridSize + (m_block.x - 1)) / m_block.x, 1, 1);

        Log::Debug("m_seedGrid: [%i, %i, %i]\n", m_seedGrid.x, m_seedGrid.y, m_seedGrid.z);
        Log::Debug("m_reduceGrid: [%i, %i, %i]\n", m_reduceGrid.x, m_reduceGrid.y, m_reduceGrid.z);

        m_frameIdx = 0;
    }

    __host__ void Host::LightProbeCamera::ClearRenderState()
    {
        m_hostAccumBuffer->Clear(vec4(0.0f));
        m_hostCompressedRayBuffer->Clear(Cuda::CompressedRay());
        //m_hostPixelFlagsBuffer->Clear(0);
    }

    __global__ void KernelSeedRayBuffer(Device::LightProbeCamera* camera, const int frameIdx)
    {
        camera->SeedRayBuffer(frameIdx);
    }

    __host__ void Host::LightProbeCamera::OnPreRenderPass(const float wallTime, const uint frameIdx)
    {
        m_frameIdx = frameIdx;
        
        KernelSeedRayBuffer << < m_seedGrid, m_block, 0, m_hostStream >> > (cu_deviceData, frameIdx);
    }

    __global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::LightProbeCamera* camera)
    {
        //if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

        camera->Composite(kKernelPos<ivec2>(), deviceOutputImage);
    }

    __host__ void Host::LightProbeCamera::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize = dim3(16, 16, 1);
        dim3 gridSize(kAccumBufferWidth / 16, kAccumBufferHeight / 16, 1);
        
        hostOutputImage->SignalSetWrite(m_hostStream);
        KernelComposite << < gridSize, blockSize, 0, m_hostStream >> > (hostOutputImage->GetDeviceInstance(), cu_deviceData);
        hostOutputImage->SignalUnsetWrite(m_hostStream);
    }

    __global__ void KernelReduceAccumulationBuffer(Device::LightProbeCamera* camera, const uint reduceBatchSize, const uvec2 batchRange)
    {
        //if (kKernelIdx == 0) printf("%u [%u, %u]\n", reduceBatchSize, batchRange.x, batchRange.y);
        
        camera->ReduceAccumulationBuffer(reduceBatchSize, batchRange);
    }

    __global__ void KernelGetProbeMinMaxSampleCount(Device::LightProbeCamera* camera, vec2* minSampleCount)
    {
        *minSampleCount = camera->GetProbeMinMaxSampleCount();
    }

    __host__ vec2 Host::LightProbeCamera::GetProbeMinMaxSampleCount() const
    {
        vec2* cu_minMax;
        vec2 minMax;
        IsOk(cudaMalloc(&cu_minMax, sizeof(vec2)));

        KernelGetProbeMinMaxSampleCount << <1, 256, 0, m_hostStream >> > (cu_deviceData, cu_minMax);
        IsOk(cudaStreamSynchronize(m_hostStream));

        IsOk(cudaMemcpy(&minMax, cu_minMax, sizeof(vec2), cudaMemcpyDeviceToHost));
        IsOk(cudaFree(cu_minMax));

        return minMax;
    }

    __host__ float Host::LightProbeCamera::GetBakeProgress() const
    {
        // FIXME: This is a horrible hack to prevent having to manually scan the accumulation buffer every frame.
        /*if (m_frameIdx < m_params.maxSamplesPerProbe)
        {
            return clamp(m_frameIdx / float(m_params.maxSamplesPerProbe), 0.0f, 1.0f);
        }*/
        
        vec2 minMax = GetProbeMinMaxSampleCount();        
        return clamp((minMax.x + 1.0f) / float(m_params.maxSamplesPerProbe), 0.0f, 1.0f);
    }

    __host__ bool Host::LightProbeCamera::ExportProbeGrid(const std::string& usdExportPath, const bool exportToUSD)
    {
        if (!m_hostLightProbeGrid->IsValid() || m_exporterState != kArmed) { return false; }          

        if (exportToUSD)
        {
            Log::Debug("Exporting to '%s'...\n", usdExportPath);
            try
            {
                USDIO::ExportLightProbeGrid(m_hostLightProbeGrid, usdExportPath);
            }
            catch (const std::runtime_error& err)
            {
                Log::Error("Error: %s\n", err.what());
            }
        }
        else
        {
            Log::Warning("Warning: Skipped USD export to '%s' because setting was not enabled.\n", usdExportPath);
        }

        m_exporterState = kFired;
        return true;
    }

    __host__ void Host::LightProbeCamera::SetLightProbeCameraParams(const LightProbeCameraParams& params)
    {
        m_params = params;
        Prepare();
    }

    __host__ void Host::LightProbeCamera::OnPostRenderPass()
    {
        // Used when parallel reducing the accumluation buffer
        uint reduceBatchSizePow2 = NearestPow2Ceil(m_params.subsamplesPerProbe);
        
        // Reduce until the batch range is equal to the size of the block
        uint batchSize = reduceBatchSizePow2;
        while (batchSize > 1)
        {
            KernelReduceAccumulationBuffer << < m_reduceGrid, m_block, 0, m_hostStream >> > (cu_deviceData, reduceBatchSizePow2, uvec2(batchSize, batchSize >> 1));
            batchSize >>= 1;
        }
        // Reduce the block in a single operation
        //KernelReduceAccumulationBuffer << < m_reduceGrid, m_block, 0, m_hostStream >> > (cu_deviceData, reduceBatchSizePow2, uvec2(batchSize, 2));

        //const vec2 minMax = GetProbeMinMaxSampleCount();
        //Log::Debug("Samples: %i\n", minMax.x);

        IsOk(cudaStreamSynchronize(m_hostStream));
    }
}