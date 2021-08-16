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
        maxSamples = 0;
    }

    __host__ LightProbeCameraParams::LightProbeCameraParams(const ::Json::Node& node) :
        LightProbeCameraParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void LightProbeCameraParams::ToJson(::Json::Node& node) const
    {
        grid.ToJson(node);
        camera.ToJson(node);

        node.AddValue("maxSamples", maxSamples);
    }

    __host__ void LightProbeCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        grid.FromJson(node, flags);
        camera.FromJson(node, flags);

        node.GetValue("maxSamples", maxSamples, flags);        
    }   

    __device__ Device::LightProbeCamera::LightProbeCamera() {  }

    __device__ void Device::LightProbeCamera::Synchronise(const LightProbeCameraParams& params)
    {
        m_params = params;
        if (m_params.maxSamples == 0) { m_params.maxSamples = INT_MAX; }

        Prepare();
    }
    __device__ void Device::LightProbeCamera::Synchronise(const Objects& objects)
    {
        m_objects = objects;
    }

    __device__ void Device::LightProbeCamera::SeedRayBuffer(const int frameIdx)
    {
        CudaDeviceAssert(kKernelIdx < 512 * 512);
        
        CompressedRay& compressedRay = (*m_objects.renderState.cu_compressedRayBuffer)[kKernelIdx];

        if (kKernelIdx > m_params.totalBuckets) 
        {
            compressedRay.Kill();
            return; 
        }

        if (!compressedRay.IsAlive() && compressedRay.sampleIdx < m_params.maxSamplesPerBucket)
        {
            //(*m_objects.cu_accumBuffer)[kKernelIdx] = 0.0f;
            CreateRay(kKernelIdx, compressedRay, frameIdx);
        }
    }

    __device__ void Device::LightProbeCamera::Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const
    {
        if (!m_objects.cu_accumBuffer || viewportPos.x >= deviceOutputImage->Width() || viewportPos.y >= deviceOutputImage->Height() ||
            viewportPos.x >= 512 || viewportPos.y >= 512) {
            return;
        }

        const auto& texel = (*m_objects.cu_accumBuffer)[viewportPos.y * 512 + viewportPos.x];

        /*int probeIdx, coeffIdx;
        ivec3 gridIdx;
        int accumIdx = kKernelY * 512 + kKernelX;
        GetProbeAttributesFromIndex(accumIdx, probeIdx, coeffIdx, gridIdx);
        deviceOutputImage->At(viewportPos) = vec4(vec3(float(coeffIdx) / 3.0f), 1.0f);
        return;*/

        // Normalise and gamma correct
        const vec3 rgb = texel.xyz / fmax(1.0f, texel.w);       
        deviceOutputImage->At(viewportPos) = vec4(rgb, 1.0f);
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

    __device__ bool RectilinearViewportToCartesian(const ivec2& viewportPos, vec3& cart)
    {
        cart = PolarToCartesian(vec2(kPi * (512 - viewportPos.y) / 512.0f, kTwoPi * viewportPos.x / 512.0f));
        return true;
    }

    __device__ void Device::LightProbeCamera::CreateRay(const uint& accumIdx, CompressedRay& ray, const int frameIdx) const
    {
        // Update the ray with the new properties and generate a random sampler from it
        // FIXME: This is horribly inefficient. Make it better.
        ray.accumIdx = accumIdx / m_params.coefficientsPerProbe;
        ray.sampleIdx++;
        ray.depth = 1;
        RNG rng(ray);

        const vec2 xi = rng.Rand<0, 1>();
        ray.od.d = SampleUnitSphere(xi);        
        
        /*float theta = kTwoPi * float(frameIdx) / 1000.0f;
        ray.od.d = vec3(0.0f, cos(theta), sin(theta));
        if (kKernelIdx == 0) {
            printf("%f\n", fmodf(float(frameIdx) / 1000.0f, 1.0f));
        }*/

        // For i'th probe, coefficients are packed together:
        // [ [ C0(i), ..., C0(i) ] [ C1(i), ..., C1(i) ], ... , [ Cn(i), ..., Cn(i)] ]

        int probeIdx, coeffIdx;
        ivec3 gridIdx;
        GetProbeAttributesFromIndex(accumIdx, probeIdx, coeffIdx, gridIdx);

        CudaDeviceAssert(accumIdx < 512 * 512);

        // Project this direction into SH and pre-normalise
        ray.accumIdx = accumIdx;
        ray.weight = kOne * SH::Project(ray.od.d, coeffIdx) * kFourPi;
        ray.depth = 2;
        ray.flags = kRayLightProbe | kRayIndirectSample;
        ray.od.o = m_params.grid.transform.PointToWorldSpace(vec3(gridIdx) / vec3(m_params.grid.gridDensity - 1) - vec3(0.5f));
    }

    __device__ void Device::LightProbeCamera::Accumulate(RenderCtx& ctx, const HitCtx& hitCtx, const vec3& value)
    {        
        //CudaDeviceAssert(ctx.emplacedRay.accumIdx < 512 * 512);
        if (ctx.emplacedRay.accumIdx >= 512 * 512)
        {
            //CudaPrintVar(ctx.emplacedRay.accumIdx, u);
            return;
        }
        
        int probeIdx, coeffIdx;
        ivec3 gridIdx;
        GetProbeAttributesFromIndex(ctx.emplacedRay.accumIdx, probeIdx, coeffIdx, gridIdx);       
        auto& texel = (*m_objects.cu_accumBuffer)[ctx.emplacedRay.accumIdx];
        
        // Accumualte SH coefficients
        if (coeffIdx < m_params.coefficientsPerProbe - 1)
        {
            texel.xyz += value;
            if (!ctx.emplacedRay.IsAlive()) { texel.w += 1.0f; }
        }
        // Accumulate probe validity
        else if(ctx.depth == 2)
        {
            // A probe sample is invalid if, on the first hit, it intersects with a back-facing surface
            float validity = float(!hitCtx.backfacing);
            texel.x += validity;
            texel.w += 1.0f;
        }
    }
    __device__ void Device::LightProbeCamera::ReduceAccumulatedSample(vec4& dest, const vec4& source)
    {              
        if (int(dest.w) >= m_params.maxSamples - 1) 
        {             
            return;
        }
        
        if (int(dest.w + source.w) < m_params.maxSamples)
        {           
            dest += source;
            return;
        }

        dest += source * (m_params.maxSamples - dest.w) / source.w;
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

        const int probeIdx = kKernelIdx / m_params.bucketsPerProbe;
        const int coeffIdx = (kKernelIdx / m_params.bucketsPerCoefficient) % m_params.coefficientsPerProbe;

        for (uint iterationSize = batchRange[0] / 2; iterationSize > batchRange[1] / 2; iterationSize >>= 1)
        {
            if (subsampleIdx < iterationSize)
            {
                // For the first iteration, copy the data out of the accumulation buffer
                if (iterationSize == batchSize / 2)
                {
                    auto& texel = reduceBuffer[kKernelIdx];
                    texel = 0.0f;
                    ReduceAccumulatedSample(texel, accumBuffer[kKernelIdx]);

                    if (subsampleIdx + iterationSize < m_params.bucketsPerCoefficient)
                    {
                        CudaDeviceAssert(kKernelIdx + iterationSize < 512 * 512);
                        //if (probeIdx == 0 && coeffIdx == 0) { printf("%i: %f + %f = %f\n", iterationSize, texel.w, accumBuffer[kKernelIdx + iterationSize].w, texel.w + accumBuffer[kKernelIdx + iterationSize].w); }
                        ReduceAccumulatedSample(texel, accumBuffer[kKernelIdx + iterationSize]);
                        
                    }
                    //else
                    //    if (probeIdx == 0 && coeffIdx == 0) { printf("%i: %f\n", iterationSize, texel.w); }
                }
                else
                {
                    CudaDeviceAssert(kKernelIdx + iterationSize < 512 * 512);
                    CudaDeviceAssert(subsampleIdx + iterationSize < m_params.bucketsPerCoefficient);
                    //if (probeIdx == 0 && coeffIdx == 0) { printf("%i: %f + %f = %f\n", iterationSize, reduceBuffer[kKernelIdx].w, reduceBuffer[kKernelIdx + iterationSize].w, reduceBuffer[kKernelIdx].w + reduceBuffer[kKernelIdx + iterationSize].w); }
                    ReduceAccumulatedSample(reduceBuffer[kKernelIdx], reduceBuffer[kKernelIdx + iterationSize]);
                }
            }
            else
            {
                //reduceBuffer[kKernelIdx] = 1.0f;
            }

            __syncthreads();
        } 

        // After the last operation, cache the accumulated value in the probe grid
        if (subsampleIdx == 0 && batchRange[0] == 2)
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
        m_params.coefficientsPerProbe = SH::GetNumCoefficients(m_params.grid.shOrder) + 1;
        // Number of light probes in the grid
        m_params.numProbes = Volume(m_params.grid.gridDensity);
        // Number of sample buckets per probe
        m_params.bucketsPerProbe = m_hostAccumBuffer->Size() / m_params.numProbes;
        // Number of sample buckets per SH coefficient per probe
        m_params.bucketsPerCoefficient = /*NearestPow2Floor*/(m_params.bucketsPerProbe / m_params.coefficientsPerProbe);
        // The maximum number of samples per bucket based on the number of buckets per coefficient
        m_params.maxSamplesPerBucket = (m_params.maxSamples == 0) ? 
                                        std::numeric_limits<int>::max() : int(1.0f + float(m_params.maxSamples) / float(m_params.bucketsPerCoefficient));

        // Adjust values so everything packs correctly
        m_params.bucketsPerProbe = m_params.bucketsPerCoefficient * m_params.coefficientsPerProbe;
        m_params.totalBuckets = m_params.bucketsPerProbe * m_params.numProbes;   

        // Used when parallel reducing the accumluation buffer
        uint reduceBatchSizePow2 = NearestPow2Ceil(m_params.bucketsPerCoefficient);

        Log::Debug("coefficientsPerProbe: %i\n", m_params.coefficientsPerProbe);
        Log::Debug("numProbes: %i\n", m_params.numProbes);
        Log::Debug("bucketsPerProbe: %i\n", m_params.bucketsPerProbe);
        Log::Debug("bucketsPerCoefficient: %i\n", m_params.bucketsPerCoefficient);
        Log::Debug("bucketsPerProbe: %i\n", m_params.bucketsPerProbe);
        Log::Debug("totalBuckets: %i\n", m_params.totalBuckets);
        Log::Debug("maxSamplesPerBucket: %i\n", m_params.maxSamplesPerBucket);
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

    __global__ void KernelSeedRayBuffer(Device::LightProbeCamera* camera, const int frameIdx)
    {
        camera->SeedRayBuffer(frameIdx);
    }

    __host__ void Host::LightProbeCamera::OnPreRenderPass(const float wallTime, const float frameIdx)
    {
        KernelSeedRayBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData, int(frameIdx));
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
        //if (kKernelIdx == 0) printf("%u [%u, %u]\n", reduceBatchSize, batchRange.x, batchRange.y);
        
        camera->ReduceAccumulationBuffer(reduceBatchSize, batchRange);
    }

    __global__ void KernelGetProbeMinMaxSampleCount(Device::LightProbeCamera* camera, vec2* minSampleCount)
    {
        *minSampleCount = camera->GetProbeMinMaxSampleCount();
    }

    __host__ void Host::LightProbeCamera::OnPostRenderPass()
    {
        // Used when parallel reducing the accumluation buffer
        uint reduceBatchSizePow2 = NearestPow2Ceil(m_params.bucketsPerCoefficient);
        
        // Reduce until the batch range is equal to the size of the block
        uint batchSize = reduceBatchSizePow2;
        while (batchSize > 1)
        {
            KernelReduceAccumulationBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData, reduceBatchSizePow2, uvec2(batchSize, batchSize >> 1));
            batchSize >>= 1;
        }
        // Reduce the block in a single operation
        //KernelReduceAccumulationBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData, reduceBatchSizePow2, uvec2(batchSize, 2));
         
        vec2* cu_minSampleCount;
        vec2 minSampleCount;
        IsOk(cudaMalloc(&cu_minSampleCount, sizeof(vec2)));

        KernelGetProbeMinMaxSampleCount << <1, 256, 0, m_hostStream >> > (cu_deviceData, cu_minSampleCount);
        IsOk(cudaStreamSynchronize(m_hostStream));
        
        IsOk(cudaMemcpy(&minSampleCount, cu_minSampleCount, sizeof(vec2), cudaMemcpyDeviceToHost));
        IsOk(cudaFree(cu_minSampleCount));
        Log::Debug("Min: %f, Max: %i\n", minSampleCount.x, minSampleCount.y);

        IsOk(cudaStreamSynchronize(m_hostStream));
    }
}