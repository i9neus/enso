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

#define kRayBufferSize          (512u * 512u * 2u)
#define kRayBufferNumBuckets    (512u * 512u)

namespace Cuda
{
    __host__ __device__ LightProbeCameraParams::LightProbeCameraParams()
    {
        lightingMode = kBakeLightingCombined;
        traversalMode = kBakeTraversalLinear;
        gridUpdateInterval = 10;
        minViableValidity = 0.0f;
    }

    __host__ LightProbeCameraParams::LightProbeCameraParams(const ::Json::Node& node, const uint flags) :
        LightProbeCameraParams()
    {
        FromJson(node, flags);
    }

    __host__ void LightProbeCameraParams::ToJson(::Json::Node& node) const
    {
        auto gridNode = node.AddChildObject("grid");
        grid.ToJson(gridNode);
        camera.ToJson(node);

        node.AddEnumeratedParameter("lightingMode", std::vector<std::string>({ "combined", "separated" }), lightingMode);
        node.AddEnumeratedParameter("traversalMode", std::vector<std::string>({ "linear", "hilbert" }), traversalMode);
        node.AddValue("gridUpdateInterval", gridUpdateInterval);
        node.AddValue("minViableValidity", minViableValidity);
    }

    __host__ void LightProbeCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        auto gridNode = node.GetChildObject("grid", flags);
        grid.FromJson(gridNode, flags);
        camera.FromJson(node, flags);

        node.GetEnumeratedParameter("lightingMode", std::vector<std::string>({ "combined", "separated" }), lightingMode, flags);
        node.GetEnumeratedParameter("traversalMode", std::vector<std::string>({ "linear", "hilbert" }), traversalMode, flags);
        node.GetValue("gridUpdateInterval", gridUpdateInterval, flags);
        node.GetValue("minViableValidity", minViableValidity, flags);
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
        assert(kKernelIdx * 2 < kRayBufferSize);

        int probeIdx = kKernelIdx / m_params.subsamplesPerProbe;
        if (probeIdx >= m_params.grid.numProbes) { return; }

        // FIXME: This asserts, but it shouldn't. Find out why.
        //assert(probeIdx < m_params.grid.numProbes);

        // Apply indirection
        if (m_params.traversalMode != kBakeTraversalLinear)
        {
            probeIdx = (*m_objects.cu_hilbertBuffer)[probeIdx];
        }
        
        // If adaptive sampling is enabled and the probe is converged, don't spawn any more rays.
        if (m_params.camera.samplingMode != kCameraSamplingFixed && (*m_objects.cu_adaptiveSamplingGrid)[probeIdx] == 0) { return; }
        
        CompressedRay* compressedRays = &(*m_objects.renderState.cu_compressedRayBuffer)[kKernelIdx * 2];

        if (kKernelIdx > m_params.totalBuckets) 
        {
            compressedRays[0].Kill();
            compressedRays[1].Kill();
            return;
        }

        // On the first frame, reset the ray and the sample index
        if (frameIdx == 0)
        {
            compressedRays[0].Reset();
            compressedRays[1].Reset();
            compressedRays[0].sampleIdx = m_seedOffset;
        }

        if (!compressedRays[0].IsAlive() && !compressedRays[1].IsAlive() &&
            (m_params.camera.maxSamples <= 0 || int(compressedRays[0].sampleIdx - m_seedOffset) < m_params.maxSamplesPerBucket))
        {
            CreateRays(probeIdx, kKernelIdx % m_params.subsamplesPerProbe, compressedRays, frameIdx);
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

        int idx = accumPos.x / (kAccumBufferWidth / 2);

        // Normalise and gamma correct
        const auto& texel = (*m_objects.cu_accumBuffers[idx])[accumPos.y * kAccumBufferWidth + accumPos.x];
        const vec3 rgb = texel.xyz / fmax(1.0f, texel.w);       
        deviceOutputImage->At(viewportPos) = vec4(rgb, 1.0f);
    }

    __device__ void Device::LightProbeCamera::Prepare()
    {
        assert(m_objects.cu_accumBuffers[0] && m_objects.cu_accumBuffers[1]);
        assert(m_objects.cu_probeGrids[0] && m_objects.cu_probeGrids[1]);

        // Only use the lower 31 bits for the seed because we need to deduce the actual sample count from it
        m_seedOffset = HashOf(uint(m_params.camera.seed) & ((1u << 31) - 1u));
    }

    __device__ void Device::LightProbeCamera::CreateRays(const int& probeIdx, const int& subsampleIdx, CompressedRay* rays, const int frameIdx) const
    {        
        //const int probeIdx = kKernelIdx / m_params.subsamplesPerProbe;
        const ivec3 gridIdx = GridPosFromProbeIdx(probeIdx, m_params.grid.gridDensity);
        const uint accumIdx = probeIdx * m_params.subsamplesPerProbe + subsampleIdx;

        auto& primary = rays[0]; 
        auto& secondary = rays[1];
        
        primary.accumIdx = accumIdx;
        primary.sampleIdx++;
        primary.depth = 0;
        RNG rng(primary);

        const vec2 xi = rng.Rand<0, 1>();
        primary.od.d = SampleUnitSphere(xi);
        primary.probeDir = primary.od.d;

        // Probes data is grouped as follows:
        // Subsample    = [SH data] [Auxilliary data]
        // Probe        = [Subsample 1] .... [Subsample N]
        // Grid         = [Probe 1] ... [Probe M]        

        // Project this direction into SH and pre-normalise
        primary.weight = kOne;
        primary.depth = 1;
        primary.flags = kRayIndirectSample | kRayLightProbe;

        primary.od.o = m_params.grid.aspectRatio * vec3(gridIdx) / vec3(m_params.grid.gridDensity - 1) - vec3(0.5f);
        primary.od.o = m_params.grid.transform.PointToWorldSpace(primary.od.o);

        secondary.accumIdx = accumIdx;
        secondary.probeDir = primary.probeDir;
        secondary.sampleIdx = primary.sampleIdx;
    }

    __device__ void Device::LightProbeCamera::Accumulate(const RenderCtx& ctx, const Ray& incidentRay, const HitCtx& hitCtx, const vec3& value, const bool isAlive)
    {          
        auto& emplacedRay = ctx.emplacedRay[0];
        assert(emplacedRay.accumIdx < kAccumBufferSize);
        
        vec3 L = value;
        if (m_params.camera.splatClamp > 0.0)
        {
            const float intensity = cwiseMax(L);
            if (intensity > m_params.camera.splatClamp)
            {
                L *= m_params.camera.splatClamp / intensity;
            }
        }
  
        // Loop through the direct and indirect accumulation buffers
        for (int gridIdx = 0; gridIdx < kLightProbeNumBuffers; ++gridIdx)
        {
            if (!m_objects.cu_accumBuffers[gridIdx]) { continue; }

            // Don't write into the indirect grid at all when in combined lighting mode
            if (m_params.lightingMode == kBakeLightingCombined && gridIdx == kLightProbeBufferIndirect) { continue; }
            
            int accumIdx = emplacedRay.accumIdx * m_params.grid.coefficientsPerProbe;
            auto& accumBuffer = *(m_objects.cu_accumBuffers[gridIdx]);
            const float weight = !isAlive;
            
            // Should we accumulate this sample?
            bool accumulate;
            switch (gridIdx)
            {
            case kLightProbeBufferDirect:
                accumulate = m_params.lightingMode == kBakeLightingCombined || incidentRay.depth <= 1; break;
            case kLightProbeBufferIndirect:
                accumulate = m_params.lightingMode == kBakeLightingSeparated && incidentRay.depth > 1; break;
            default: // kLightProbeBufferHalf
                // NOTE: Half buffer gets all types of samples regardless of what lighting mode we're using
                accumulate = emplacedRay.sampleIdx % 2 == 1;
            }

            if(accumulate)  
            {              
                // Project and accumulate the SH coefficients
                for (int shIdx = 0; shIdx < m_params.grid.coefficientsPerProbe - 1; ++shIdx, ++accumIdx)
                {
                    accumBuffer[accumIdx] += vec4(L * SH::Project(ctx.emplacedRay[0].probeDir, shIdx) * kFourPi, weight);
                }
            }
            else
            {
                // Just increment the weights
                for (int shIdx = 0; shIdx < m_params.grid.coefficientsPerProbe - 1; ++shIdx, ++accumIdx) { accumBuffer[accumIdx][3] += weight; }
            }

            if (gridIdx != kLightProbeBufferHalf && incidentRay.IsIndirectSample())//&& incidentRay.depth == 1)
            {
                // Accumulate validity and mean distance
                // A probe sample is valid if, on the first hit, it intersects with a front-facing surface or it leaves the scene
                accumBuffer[accumIdx] += vec4(float(!hitCtx.isValid || !hitCtx.backfacing), 
                                              //1.0f / max(1e-10f, incidentRay.tNear),
                                              min(m_params.grid.transform.scale().x, incidentRay.tNear) / m_params.grid.transform.scale().x,
                                              0.0f, 1.0f);
            }
        }
    }
    __device__ void Device::LightProbeCamera::ReduceAccumulatedSample(vec4& dest, const vec4& source)
    {              
        if (int(dest.w) >= m_params.grid.maxSamplesPerProbe - 1) 
        {             
            return;
        }
        
        if (int(dest.w + source.w) < m_params.grid.maxSamplesPerProbe)
        {           
            dest += source;
            return;
        }

        dest += source * (m_params.grid.maxSamplesPerProbe - dest.w) / source.w;
    }

    __device__ void Device::LightProbeCamera::ReduceAccumulationBuffer(Device::Array<vec4>* cu_accumBuffer, Device::LightProbeGrid* cu_probeGrid, const uint batchSize, const uvec2 batchRange)
    {         
        if (kKernelIdx >= m_params.totalBuckets) { return; }

        assert(cu_accumBuffer);
        assert(cu_probeGrid);
        assert(m_objects.cu_reduceBuffer);

        //if (batchRange[0] == batchSize) (*m_objects.cu_reduceBuffer)[kKernelIdx] = 0.0f;

        auto& accumBuffer = *cu_accumBuffer;
        auto& reduceBuffer = *m_objects.cu_reduceBuffer;
        
        const int probeIdx = kKernelIdx / m_params.bucketsPerProbe;
        const int probeSubsampleIdx = (kKernelIdx / m_params.grid.coefficientsPerProbe) % m_params.subsamplesPerProbe;
        const int coeffIdx = kKernelIdx % m_params.grid.coefficientsPerProbe;

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
                        assert(kKernelIdx + iterationSize * m_params.grid.coefficientsPerProbe < kAccumBufferSize);
                        //if (probeIdx == 0 && coeffIdx == 0) { printf("%i: %f + %f = %f\n", iterationSize, texel.w, accumBuffer[kKernelIdx + iterationSize].w, texel.w + accumBuffer[kKernelIdx + iterationSize].w); }
                        ReduceAccumulatedSample(texel, accumBuffer[kKernelIdx + iterationSize * m_params.grid.coefficientsPerProbe]);

                    }
                    //else
                    //    if (probeIdx == 0 && coeffIdx == 0) { printf("%i: %f\n", iterationSize, texel.w); }
                }
                else
                {
                    assert(kKernelIdx + iterationSize * m_params.grid.coefficientsPerProbe < kAccumBufferSize);
                    assert(probeSubsampleIdx + iterationSize < m_params.subsamplesPerProbe);
                    //if (probeIdx == 0 && coeffIdx == 0) { printf("%i: %f + %f = %f\n", iterationSize, reduceBuffer[kKernelIdx].w, reduceBuffer[kKernelIdx + iterationSize].w, reduceBuffer[kKernelIdx].w + reduceBuffer[kKernelIdx + iterationSize].w); }
                    ReduceAccumulatedSample(reduceBuffer[kKernelIdx], reduceBuffer[kKernelIdx + iterationSize * m_params.grid.coefficientsPerProbe]);
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
            //const int coeffIdx = (kKernelIdx / m_params.bucketsPerCoefficient) % m_params.grid.coefficientsPerProbe;
            if (coeffIdx == m_params.grid.coefficientsPerProbe - 1)
            {
               const float norm = max(1.0f, texel.w);

               texel.x /= norm;                         // Probe validity
               //texel.y = norm / max(1e-10f, texel.y); // Harmonic mean distance
               texel.y /= norm;                         // Geometric mean distance
               texel.z = texel.w;                       // Store the total number of samples
            }
            else
            {
                texel /= max(1.0f, texel.w);
            }

            cu_probeGrid->SetSHCoefficient(probeIdx, coeffIdx, texel.xyz);
        }
    }

    __device__ void Device::LightProbeCamera::BuildLightProbeErrorGrid()
    {
        if (kKernelIdx >= m_params.grid.numProbes) { return; }
        assert(m_objects.cu_lightProbeErrorGrids[0]);

        const vec3* P0 = m_objects.cu_probeGrids[0]->At(kKernelIdx);
        const vec3* P1 = m_objects.cu_probeGrids[1]->At(kKernelIdx);
        const vec3* PHalf = m_objects.cu_probeGrids[2]->At(kKernelIdx);       

        // Go channel by channel to find the peak irradiance
        float peakSqrErrorI = 0.0f;
        float peakI = 0.0f;
        for (int chnlIdx = 0; chnlIdx < 3; ++chnlIdx)
        {
            // Load the coefficients from the first map
            float L0 = P0[0][chnlIdx];
            vec3 L1(P0[1][chnlIdx], P0[2][chnlIdx], P0[3][chnlIdx]);

            // Add the coefficients from the second map if we're in separated mode
            if (m_params.lightingMode == kBakeLightingSeparated)
            {
                L0 += P1[0][chnlIdx];
                L1 += vec3(P1[1][chnlIdx], P1[2][chnlIdx], P1[3][chnlIdx]);
            }

            // Load the coefficients from the third map
            const float L0Half = PHalf[0][chnlIdx];
            const vec3 L1Half(PHalf[1][chnlIdx], PHalf[2][chnlIdx], PHalf[3][chnlIdx]);

            // Estimate the peak irradiance over the unit sphere
            float M = L0Half * SH::GetLegendreCoefficient(0) + length(L1Half) * SH::GetLegendreCoefficient(1);
            float N = (L0 - L0Half) * SH::GetLegendreCoefficient(0) + (length(L1) - length(L1Half)) * SH::GetLegendreCoefficient(1);

            // Gamma ramp
            if (m_params.camera.adaptiveSamplingGamma != 1.0f)
            {
                M = powf(M, 1 / m_params.camera.adaptiveSamplingGamma);
                N = powf(N, 1 / m_params.camera.adaptiveSamplingGamma);
            }

            // Update the peak irradiance and error over all channels
            peakSqrErrorI = max(peakSqrErrorI, sqr(M - N) * 2.0f);
            peakI = max(peakI, (M + N) * 0.5f);
        }

        // Record data in the grid
        auto& probe = (*m_objects.cu_lightProbeErrorGrids[0])[kKernelIdx];
        probe.x = peakI;
        probe.y = peakSqrErrorI;
    }

    __device__ void Device::LightProbeCamera::DilateLightProbeErrorGrid()
    {
        if (kKernelIdx >= m_params.grid.numProbes) { return; }
        assert(m_objects.cu_lightProbeErrorGrids[0]);

        // Create validity and edge masks to save time later on.
        const ivec3 gridPos0 = GridPosFromProbeIdx(kKernelIdx, m_params.grid.gridDensity);
        vec2 peakProbe(0.0f);
        for (int z = -1, idx = 0; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++, idx++)
                {
                    const ivec3 gridPosK = gridPos0 + ivec3(x, y, z);
                    if (gridPosK.x < 0 || gridPosK.x >= m_params.grid.gridDensity.x ||
                        gridPosK.y < 0 || gridPosK.y >= m_params.grid.gridDensity.y ||
                        gridPosK.z < 0 || gridPosK.z >= m_params.grid.gridDensity.z)
                    {                
                        continue;
                    }
    
                    // The dilated error is simply the maximum value of its neighbours
                    const auto& probe = (*m_objects.cu_lightProbeErrorGrids[0])[ProbeIdxFromGridPos(gridPosK, m_params.grid.gridDensity)];
                    if (probe.y > peakProbe.y)
                    {
                        peakProbe = probe;
                    }
                }
            }
        }

        (*m_objects.cu_lightProbeErrorGrids[1])[kKernelIdx] = peakProbe;
    }

    __device__ void Device::LightProbeCamera::ReduceLightProbeErrorData(LightProbeCameraAggregateStatistics& stats)
    {
        assert(m_objects.cu_lightProbeErrorGrids[0]);
        assert(m_objects.cu_adaptiveSamplingGrid);
        assert(m_objects.cu_meanI);
        
        constexpr float kMinMSENorm = 1e-10f;
        
        __shared__ int localNumActiveProbes[256], numActiveProbes;
        __shared__ vec2 localI[256], I;

        const int startIdx = m_params.grid.numProbes * kKernelIdx / 256;
        const int endIdx = m_params.grid.numProbes * (kKernelIdx + 1) / 256;

        // Sum peak irradiance over the grid
        localNumActiveProbes[kKernelIdx] = 0;
        localI[kKernelIdx] = vec2(0.0f);
        for (int idx = startIdx; idx < endIdx; idx++)
        {          
            localI[kKernelIdx] += (*m_objects.cu_lightProbeErrorGrids[0])[idx];
        }
        localI[kKernelIdx] /= max(1, endIdx - startIdx);

        __syncthreads();

        if (kKernelIdx == 0)
        {       
            I = vec2(0.0f);
            for (int idx = 0; idx < 256; idx++) { I += localI[idx]; }
            I /= 256.0f;

            // Update the MSE value and normalise it if necessary
            stats.meanI =  I.x;
            *m_objects.cu_meanI = I.x;
            stats.MSE = I.y;
            if (m_params.camera.samplingMode == kCameraSamplingAdaptiveRelative)
            {
                stats.MSE /= max(kMinMSENorm, I.x);
            }
        }

        __syncthreads();

        // Populate the adaptive sampling grid
        for (int i = startIdx; i < endIdx; i++)
        {
            float sqrError = (*m_objects.cu_lightProbeErrorGrids[0])[i].y;

            // Relative mode uses the mean probe irradiance as its normalisation factor. 
            // We multiply this by a factor of two assuming irradiance values are evently distributed between N and 0, so
            // the mean is half the value we need to normalise by.
            if (m_params.camera.samplingMode == kCameraSamplingAdaptiveRelative)
            {
                sqrError /= max(kMinMSENorm, sqr(I.x * 2.0f));
            }
            
            // Set the entry in the adaptive sampling grid and accumulate the number of active probes
            const bool isActive = sqrError > sqr(m_params.camera.errorThreshold);
            (*m_objects.cu_adaptiveSamplingGrid)[i] = uchar(isActive);

            localNumActiveProbes[kKernelIdx] += int(isActive);
        }

        __syncthreads();

        if (kKernelIdx == 0)
        {
            int numActiveProbes = 0;
            for (int idx = 0; idx < 256; idx++) { numActiveProbes += localNumActiveProbes[idx]; }
            
            stats.bakeConvergence = 1.0f - numActiveProbes / float(m_params.grid.numProbes);
        }
    }

    __host__ Host::LightProbeCamera::LightProbeCamera(const ::Json::Node& node, const std::string& id) :
        Host::Camera(node, id, kRayBufferSize),
        m_block(16 * 16, 1, 1),
        m_seedGrid(1, 1, 1),
        m_reduceGrid(1, 1, 1),
        m_hostMeanI(1.0f)
    {
        // Register events for deligates to watch
        RegisterEvent("OnBuildGrids");

        // TODO: This is to maintain backwards compatibility. Deprecate it when no longer required.
        m_gridIDs[0] = "grid_noisy_direct";
        m_gridIDs[1] = "grid_noisy_indirect";

        node.GetValue("gridDirectID", m_gridIDs[0], Json::kRequiredWarn | Json::kNotBlank);
        node.GetValue("gridIndirectID", m_gridIDs[1], Json::kRequiredWarn | Json::kNotBlank);
        node.GetValue("gridHalfID", m_gridIDs[2], Json::kSilent);

        // Create reduction and adaptive sampling buffers
        m_hostReduceBuffer = AssetHandle<Host::Array<vec4>>(tfm::format("%s_probeReduceBuffer", id), kAccumBufferSize, m_hostStream);
        m_hostHilbertBuffer = AssetHandle<Host::Array<uint>>(tfm::format("%s_hilbertBuffer", id), kAccumBufferSize, m_hostStream);
        m_hostLightProbeErrorGrids[0] = AssetHandle<Host::Array<vec2>>(tfm::format("%s_probeErrorGrids0", id), kAccumBufferSize, m_hostStream);
        m_hostLightProbeErrorGrids[1] = AssetHandle<Host::Array<vec2>>(tfm::format("%s_probeErrorGrids1", id), kAccumBufferSize, m_hostStream);
        m_hostAdaptiveSamplingGrid = AssetHandle<Host::Array<uchar>>(tfm::format("%s_adaptiveSamplingGrid", id), kAccumBufferSize, m_hostStream);

        // Instantiate the camera object on the device
        cu_deviceData = InstantiateOnDevice<Device::LightProbeCamera>();

        // Create the accumulation buffers and probe grids
        for (int idx = 0; idx < m_hostAccumBuffers.size(); ++idx)
        {
            // Don't create grids that don't have IDs
            if (m_gridIDs[idx].empty()) { continue; }

            m_hostAccumBuffers[idx] = AssetHandle<Host::Array<vec4>>(tfm::format("%s_probeAccumBuffer%i", id, idx), kAccumBufferSize, m_hostStream);
            m_hostAccumBuffers[idx]->Clear(vec4(0.0f));
            m_deviceObjects.cu_accumBuffers[idx] = m_hostAccumBuffers[idx]->GetDeviceInstance();

            // Create the probe grid objects and attach external buffers to them
            m_hostLightProbeGrids[idx] = AssetHandle<Host::LightProbeGrid>(m_gridIDs[idx], m_gridIDs[idx]);
            m_hostLightProbeGrids[idx]->SetExternalBuffers(m_hostAdaptiveSamplingGrid, m_hostLightProbeErrorGrids[0], m_hostMeanI);

            m_deviceObjects.cu_probeGrids[idx] = m_hostLightProbeGrids[idx]->GetDeviceInstance();
        }

        // Sychronise the device objects
        m_deviceObjects.cu_reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
        m_deviceObjects.renderState.cu_compressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
        m_deviceObjects.cu_hilbertBuffer = m_hostHilbertBuffer->GetDeviceInstance();
        m_deviceObjects.renderState.cu_blockRayOccupancy = m_hostBlockRayOccupancy->GetDeviceInstance();
        m_deviceObjects.renderState.cu_renderStats = m_hostRenderStats->GetDeviceInstance();
        m_deviceObjects.cu_lightProbeErrorGrids[0] = m_hostLightProbeErrorGrids[0]->GetDeviceInstance();
        m_deviceObjects.cu_lightProbeErrorGrids[1] = m_hostLightProbeErrorGrids[1]->GetDeviceInstance();
        m_deviceObjects.cu_adaptiveSamplingGrid = m_hostAdaptiveSamplingGrid->GetDeviceInstance();
        m_deviceObjects.cu_meanI = m_hostMeanI.GetDeviceInstance();

        // Objects are re-synchronised at every JSON update
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ AssetHandle<Host::RenderObject> Host::LightProbeCamera::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kCamera) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LightProbeCamera(json, id), id);
    }

    __host__ void Host::LightProbeCamera::OnDestroyAsset()
    {
        Host::Camera::OnDestroyAsset();

        // Destroy the light probe grids 
        for (auto& grid : m_hostLightProbeGrids) { grid.DestroyAsset(); }

        // Destroy the rest of the objects
        for (auto& accumBuffer : m_hostAccumBuffers) { accumBuffer.DestroyAsset(); }
        for (auto& grid : m_hostLightProbeErrorGrids) { grid.DestroyAsset();  }
        m_hostReduceBuffer.DestroyAsset();
        m_hostAdaptiveSamplingGrid.DestroyAsset();

        DestroyOnDevice(cu_deviceData);
    }

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::LightProbeCamera::GetChildObjectHandles()
    {
        std::vector<AssetHandle<Host::RenderObject>> children;
        for (auto& grid : m_hostLightProbeGrids)
        {
            if (grid) { children.push_back(AssetHandle<Host::RenderObject>(grid)); }
        }
        return children;
    }

    __host__ void Host::LightProbeCamera::FromJson(const ::Json::Node& parentNode, const uint flags)
    {       
        Prepare(LightProbeCameraParams(parentNode, flags));
    }

    __host__ void Host::LightProbeCamera::GenerateHilbertBuffer(const LightProbeCameraParams& newParams)
    {
        if (newParams.grid.gridDensity == m_params.grid.gridDensity &&
            newParams.traversalMode == m_params.traversalMode) { return; }

        if (newParams.grid.numProbes == 0)
        {
            m_hostHilbertBuffer->Resize(0);
            return;
        }        
         
        /*
            Symbol table for Lindenmayer system for 3D Hilbert curve
            0 = Rotate 90 anticlockwise around X axis
            1 = Rotate 90 clockwise around X axis
            2 = Rotate 90 anticlockwise around Y axis,
            ....
            * = Recurse
            + = Forward 1 unit
        */

        static const std::string LSystem = "24*+24*+*1+255*+*3+055*+*1+5*15";

        std::array<int, 36> transformLUT;
        std::array<ivec3, 6> directionLUT;
        static const std::array<imat3, 6> rotMat = 
        {
            imat3(ivec3(0, 1, 0), ivec3(-1, 0, 0), ivec3(0, 0, 1)),
            imat3(ivec3(0, -1, 0), ivec3(1, 0, 0), ivec3(0, 0, 1)),
            imat3(ivec3(0, 0, 1), ivec3(0, 1, 0), ivec3(-1, 0, 0)),
            imat3(ivec3(0, 0, -1), ivec3(0, 1, 0), ivec3(1, 0, 0)),
            imat3(ivec3(1, 0, 0), ivec3(0, 0, 1), ivec3(0, -1, 0)),
            imat3(ivec3(1, 0, 0), ivec3(0, 0, -1), ivec3(0, 1, 0))
        }; 

        std::vector<uint> hilbertIndices;
        std::vector<int> LStack;

        // Reserve some space for the indirection buffer
        hilbertIndices.reserve(newParams.grid.numProbes);
        hilbertIndices.push_back(0);
        LStack.push_back(0);

        // Compute the size of the Hilbert cube that will completely enclose the probe grid
        int hilbertSize;
        int numIterations = 1;
        for (hilbertSize = 2; hilbertSize < cwiseMax(newParams.grid.gridDensity); hilbertSize <<= 1, ++numIterations) {}
        Assert(numIterations < 8); // Sanity check       

        ivec3 turtleP(0);
        imat3 turtleM = imat3::Indentity();

        std::vector<uchar> checksum(newParams.grid.numProbes, 0);

        while (!LStack.empty())
        {
            const char L = LSystem[LStack.back()++];
            switch (L)
            {
            case '*':
                // Recurse by pushing a new rule onto the stack
                if (LStack.size() < numIterations)
                {
                    LStack.push_back(0);
                }
                break;
            case '+':
                // Increment the position by point unit in the current direction
                turtleP += ivec3(turtleM[0][0], -turtleM[1][0], -turtleM[2][0]);

                AssertMsgFmt(!(turtleP.x < 0 || turtleP.x >= hilbertSize || turtleP.y < 0 || turtleP.y >= hilbertSize || turtleP.z < 0 || turtleP.z >= hilbertSize),
                    "Turtle went out of bounds: %s", turtleP.format().c_str());

                if (turtleP.x < newParams.grid.gridDensity.x && turtleP.y < newParams.grid.gridDensity.y && turtleP.z < newParams.grid.gridDensity.z)
                {
                    // Push the index of this voxel into the indirection buffer
                    hilbertIndices.push_back(newParams.grid.gridDensity.x * (turtleP.z * newParams.grid.gridDensity.y + turtleP.y) + turtleP.x);
                    checksum[hilbertIndices.back()]++;
                }
                break;

            default:
                // Transform the turtleP direction
                turtleM = turtleM * rotMat[int(L) - int('0')];
            }

            // If we've reached the end of the rule, pop it off the stack
            if (LStack.back() == LSystem.length()) { LStack.pop_back(); }
        }        

        AssertMsgFmt(hilbertIndices.size() == newParams.grid.numProbes, "Size mismatch: %i -> %i", hilbertIndices.size(), newParams.grid.numProbes); // Sanity check

        // Diagnostics
        int count[3] = { 0, 0, 0 };
        for (int i = 0; i < checksum.size(); ++i)
        {
            count[min(2, int(checksum[i]))]++;
        }
        Log::Error("%i -> %i, %i, %i", checksum.size(), count[0], count[1], count[2]);

        //for (int i = 0; i < newParams.grid.numProbes; ++i) { hilbertIndices[i] = i; }

        // Upload the indices to the device
        m_hostHilbertBuffer->Upload(hilbertIndices);

    }

    __host__ void Host::LightProbeCamera::Prepare(LightProbeCameraParams newParams)
    {
        newParams.grid.Prepare();

        // Reduce the size of the grid if it exceeds the size of the accumulation buffer
        const int maxNumProbes = min(kAccumBufferSize / newParams.grid.coefficientsPerProbe, kRayBufferNumBuckets);
        if (Volume(newParams.grid.gridDensity) > maxNumProbes)
        {
            const auto oldDensity = newParams.grid.gridDensity;
            while (Volume(newParams.grid.gridDensity) > maxNumProbes)
            {
                newParams.grid.gridDensity = max(ivec3(1), newParams.grid.gridDensity - ivec3(1));
            }
            Log::Error("WARNING: The size of the probe grid %s is too large for the accumulation buffer. Reducing to %s.\n", oldDensity.format(), newParams.grid.gridDensity.format());
        }

        // Prepare the light probe grid with the new parameters
        newParams.grid.camera = newParams.camera;
        for (auto& grid : m_hostLightProbeGrids)
        {
            if (grid) { grid->Prepare(newParams.grid); }
        }

        // Number of light probes in the grid
        newParams.grid.numProbes = Volume(newParams.grid.gridDensity);
        // Number of SH parameter sets per probe, reduced later to get the final value 
        newParams.subsamplesPerProbe = min(kRayBufferNumBuckets / newParams.grid.numProbes,
            kAccumBufferSize / (newParams.grid.numProbes * newParams.grid.coefficientsPerProbe));

        // The maximum number of samples per bucket based on the number of buckets per coefficient
        newParams.grid.maxSamplesPerProbe = newParams.maxSamplesPerBucket = std::numeric_limits<int>::max();
        if (newParams.camera.maxSamples > 0)
        {
            newParams.grid.maxSamplesPerProbe = newParams.camera.maxSamples;
            newParams.maxSamplesPerBucket = int(1.0f + float(newParams.camera.maxSamples) / float(newParams.subsamplesPerProbe));
        }

        // Derive some more values
        newParams.bucketsPerProbe = newParams.subsamplesPerProbe * newParams.grid.coefficientsPerProbe;
        newParams.totalBuckets = newParams.bucketsPerProbe * newParams.grid.numProbes;
        newParams.totalSubsamples = newParams.subsamplesPerProbe * newParams.grid.numProbes;

        // Used when parallel reducing the accumluation buffer
        uint reduceBatchSizePow2 = NearestPow2Ceil(newParams.subsamplesPerProbe);

        Log::Debug("coefficientsPerProbe: %i\n", newParams.grid.coefficientsPerProbe);
        Log::Debug("numProbes: %i\n", newParams.grid.numProbes);
        Log::Debug("subsamplesPerProbe: %i\n", newParams.subsamplesPerProbe);
        Log::Debug("bucketsPerProbe: %i\n", newParams.bucketsPerProbe);
        Log::Debug("totalBuckets: %i\n", newParams.totalBuckets);
        Log::Debug("maxSamplesPerBucket: %i\n", newParams.maxSamplesPerBucket);
        Log::Debug("reduceBatchSizePow2: %i\n", reduceBatchSizePow2);

        // Update the Hilbert indirect buffer if required
        GenerateHilbertBuffer(newParams);

        // Update the camera params object with the new params
        m_params = newParams;

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
        for (auto& accumBuffer : m_hostAccumBuffers)
        {
            if (accumBuffer)
            {
                accumBuffer->Clear(vec4(0.0f));
            }
        }
        m_hostCompressedRayBuffer->Clear(Cuda::CompressedRay());
        //m_hostLightProbeErrorGrids[0]->Clear(vec2(0.0f));
        m_hostAdaptiveSamplingGrid->Clear(1);
        *m_aggregateStats = LightProbeCameraAggregateStatistics();
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

    __global__ void KernelReduceAccumulationBuffer(Device::LightProbeCamera* camera, Device::Array<vec4>* cu_accumBuffer, Device::LightProbeGrid* cu_probeGrid,
        const uint reduceBatchSize, const uvec2 batchRange)
    {
        camera->ReduceAccumulationBuffer(cu_accumBuffer, cu_probeGrid, reduceBatchSize, batchRange);
    }

    __global__ void KernelReduceLightProbeErrorData(Device::LightProbeCamera* cu_camera, LightProbeCameraAggregateStatistics* cu_stats)
    {
        cu_camera->ReduceLightProbeErrorData(*cu_stats);
    }

    __host__ void Host::LightProbeCamera::UpdateProbeGridAggregateStatistics()
    {       
        // Reset the stats
        m_aggregateStats = LightProbeCameraAggregateStatistics();
        
        // If we're using adaptive sampling to monitor convergence, construct the grid
        if (m_params.camera.samplingMode != kCameraSamplingFixed)
        {
            // Reduce the adaptive sampling data to find the total number of converged probes
            KernelReduceLightProbeErrorData << <1, 256, 0, m_hostStream >> > (cu_deviceData, m_aggregateStats.GetDeviceInstance());
            IsOk(cudaStreamSynchronize(m_hostStream));

            // Sync the host copy 
            m_aggregateStats.Download();
        }

        auto& as = *m_aggregateStats;
        as.isConverged = true;
        as.bakeProgress = 0.0f;
        as.meanGridValidity = 0.0f;
        as.minMaxSamples = vec2(std::numeric_limits<float>::max(), 0.0f);
        as.numActiveGrids = 0;

        // Pull the aggreate stats from each light probe grid in turn
        for (int gridIdx = 0; gridIdx < kLightProbeNumBuffers; ++gridIdx)
        {
            auto& grid = m_hostLightProbeGrids[gridIdx];
            if (!grid) { continue; }

            const auto& gs = grid->UpdateAggregateStatistics(m_params.grid.maxSamplesPerProbe);

            // Only use the direct/indirect grids to measure statistics
            if (gridIdx < 2 && m_params.camera.maxSamples > 0)
            {
                as.numActiveGrids++;

                const float progress = clamp(std::ceil(gs.minMaxSamples.x) / float(m_params.grid.maxSamplesPerProbe), 0.0f, 1.0f);
                if (progress > 0 && (progress < as.bakeProgress || as.bakeProgress == 0.0f))
                {
                    as.bakeProgress = progress;
                }

                as.minMaxSamples[0] = min(as.minMaxSamples[0], gs.minMaxSamples[0]);
                as.minMaxSamples[1] = max(as.minMaxSamples[1], gs.minMaxSamples[1]);
                as.meanGridValidity += gs.meanValidity;
            }

            if (!gs.isConverged) { as.isConverged = false; }
        }

        as.meanGridValidity = (as.numActiveGrids == 0) ? -1.0f : (as.meanGridValidity / float(as.numActiveGrids));
    }

    __host__ const LightProbeCameraAggregateStatistics& Host::LightProbeCamera::PollBakeProgress()
    {
        // FIXME: This is a horrible hack to prevent having to manually scan the accumulation buffer every frame.
        /*if (m_frameIdx < m_params.maxSamplesPerProbe)
        {
            return clamp(m_frameIdx / float(m_params.maxSamplesPerProbe), 0.0f, 1.0f);
        }*/

        if ((m_frameIdx - 2) % m_params.gridUpdateInterval == 0)
        {
            UpdateProbeGridAggregateStatistics();
        }

        return *m_aggregateStats;
    }

    __host__ bool Host::LightProbeCamera::ExportProbeGrid(const LightProbeGridExportParams& params)
    {
        // Recompile the grids to make sure everything is included in the export
        Compile();

        for (int gridIdx = 0; gridIdx < kLightProbeNumBuffers; ++gridIdx)
        {
            // Don't write out indirect when running in combined mode
            if (m_params.lightingMode == kBakeLightingCombined && gridIdx == kLightProbeBufferIndirect) { continue; }

            // Only write grids that have valid paths associated with them
            if (gridIdx >= params.usdExportPaths.size()) { continue; }

            const auto& stats = m_hostLightProbeGrids[gridIdx]->GetAggregateStatistics();

            // If the validity is outside the valid range, all grids will be similarly invalid so bail immediatel
            if (stats.meanValidity < params.minGridValidity || stats.meanValidity > params.maxGridValidity)
            {
                Log::Warning("Warning: Cannot not export probe grid. Mean validity %f is outside valid range [%f, %f]", stats.meanValidity, params.minGridValidity, params.maxGridValidity);
                break;
            }

            // Only export to USD if explicitly flagged to do so
            if (params.exportToUSD)
            {
                Log::Debug("Exporting to '%s'...\n", params.usdExportPaths[gridIdx]);
                try
                {
                    USDIO::ExportLightProbeGrid(m_hostLightProbeGrids[gridIdx], params.usdExportPaths[gridIdx], USDIO::SHPackingFormat::kUnity);
                }
                catch (const std::runtime_error& err)
                {
                    Log::Error("Error: %s\n", err.what());
                }
            }
            else
            {
                Log::Warning("Warning: Skipped USD export to '%s' because setting was not enabled.\n", params.usdExportPaths[gridIdx]);
                break;
            }
        }

        return true;
    }

    __host__ void Host::LightProbeCamera::SetLightProbeCameraParams(const LightProbeCameraParams& newParams)
    {
        Prepare(newParams);
    }

    __host__ void Host::LightProbeCamera::BuildLightProbeGrids()
    {
        // Used when parallel reducing the accumluation buffer
        uint reduceBatchSizePow2 = NearestPow2Ceil(m_params.subsamplesPerProbe);

        for (int gridIdx = 0; gridIdx < kLightProbeNumBuffers; ++gridIdx)
        {
            if (!m_hostLightProbeGrids[gridIdx]) { continue; }

            // Indirect buffer isn't used when running in combined mode
            if (m_params.lightingMode == kBakeLightingCombined && gridIdx == kLightProbeBufferIndirect) { continue; }

            auto& grid = *m_hostLightProbeGrids[gridIdx];

            // Reduce until the batch range is equal to the size of the block
            uint batchSize = reduceBatchSizePow2;
            while (batchSize > 1)
            {
                KernelReduceAccumulationBuffer << < m_reduceGrid, m_block, 0, m_hostStream >> > (cu_deviceData, m_deviceObjects.cu_accumBuffers[gridIdx],
                    m_deviceObjects.cu_probeGrids[gridIdx],
                    reduceBatchSizePow2, uvec2(batchSize, batchSize >> 1));
                batchSize >>= 1;
            }
            // Reduce the block in a single operation
            //KernelReduceAccumulationBuffer << < m_reduceGrid, m_block, 0, m_hostStream >> > (cu_deviceData, reduceBatchSizePow2, uvec2(batchSize, 2));

            //const vec2 minMax = GetProbeGridAggregateStatistics();
            //Log::Debug("Samples: %i\n", minMax.x);

            grid.Integrate();

            IsOk(cudaStreamSynchronize(m_hostStream));
        }

        OnEvent("OnBuildGrids");
    }

    __global__ void KernelBuildLightProbeErrorGrid(Device::LightProbeCamera* cu_camera)
    {
        cu_camera->BuildLightProbeErrorGrid();
    }

    __global__ void KernelDilateLightProbeErrorGrid(Device::LightProbeCamera* cu_camera)
    {
        cu_camera->DilateLightProbeErrorGrid();
    }

    __host__ void Host::LightProbeCamera::BuildLightProbeErrorGrid()
    {
        const int gridSize = (m_params.grid.numProbes + 255) / 256;
        KernelBuildLightProbeErrorGrid << < gridSize, 256, 0, m_hostStream >> > (cu_deviceData);
        IsOk(cudaStreamSynchronize(m_hostStream));

        // Dilate the adaptive sampling grid
        KernelDilateLightProbeErrorGrid << < gridSize, 256, 0, m_hostStream >> > (cu_deviceData);
        IsOk(cudaStreamSynchronize(m_hostStream));
        m_hostLightProbeErrorGrids[0]->Swap(*m_hostLightProbeErrorGrids[1]);
    }

    __host__ void Host::LightProbeCamera::OnPostRenderPass()
    {
        if ((m_frameIdx - 2) % m_params.gridUpdateInterval == 0)
        {
            Compile();
        }
    }

    __host__ void Host::LightProbeCamera::Compile()
    {
        // Compile the data in the accumulation buffer into the grid data structures
        BuildLightProbeGrids();

        // If we're using adaptive sampling to monitor convergence, construct the grid
        if (m_params.camera.samplingMode != kCameraSamplingFixed)
        {
            BuildLightProbeErrorGrid();
        }

        // Compute aggregate statistics about each grid such as min and max sample count
        UpdateProbeGridAggregateStatistics();
    }

    __host__ bool Host::LightProbeCamera::EmitStatistics(Json::Node& rootNode) const
    {        
        rootNode.AddValue("isActive", m_params.camera.isActive);
        rootNode.AddValue("bakeProgress", m_aggregateStats->bakeProgress);
        rootNode.AddValue("bakeConvergence", m_aggregateStats->bakeConvergence);
        rootNode.AddValue("mse", m_aggregateStats->MSE);
        rootNode.AddValue("meanI", m_aggregateStats->meanI);
        rootNode.AddValue("frameIdx", m_frameIdx);

        Json::Node gridSetNode = rootNode.AddChildObject("grids");
        for (int gridIdx = 0; gridIdx < kLightProbeNumBuffers; ++gridIdx)
        {
            if (!m_hostLightProbeGrids[gridIdx]) { continue; }

            const auto& stats = m_hostLightProbeGrids[gridIdx]->GetAggregateStatistics();
            
            Json::Node gridNode = gridSetNode.AddChildObject(m_gridIDs[gridIdx]);         
            gridNode.AddValue("minSamples", int(stats.minMaxSamples.x));
            gridNode.AddValue("maxSamples", int(stats.minMaxSamples.y));
            gridNode.AddValue("meanProbeValidity", stats.meanValidity);
            gridNode.AddValue("meanProbeDistance", stats.meanDistance);

            /*std::vector<std::vector<uint>> histogramData(4);
            for (int idx = 0; idx < 4; ++idx)
            {
                histogramData[idx].resize(50);
                std::memcpy(histogramData[idx].data(), &stats.coeffHistogram[50 * idx], sizeof(uint) * 50);
            }
            gridNode.AddArray2D("coeffHistograms", histogramData);*/

            std::vector<float> peakIntensityData(4);
            std:memcpy(peakIntensityData.data(), stats.meanSqrIntensity, sizeof(float) * 4);
            gridNode.AddArray("peakIntensity", peakIntensityData);

        }

        return true;
    }
}