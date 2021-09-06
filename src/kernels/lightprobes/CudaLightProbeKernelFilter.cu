#include "CudaLightProbeKernelFilter.cuh"
#include "../CudaManagedArray.cuh"
#include "CudaFilters.cuh"

#include "generic/JsonUtils.h"

#define kBlockSize 256
#define kMaxCoefficients 5

namespace Cuda
{
    __host__ __device__ LightProbeKernelFilterParams::LightProbeKernelFilterParams() : 
        filterType(kKernelFilterGaussian),
        radius(1.0f),
        trigger(false) 
    {
        nlm.alpha = 1.0f;
        nlm.K = 1.0f;
    }
    
    __host__ void LightProbeKernelFilterParams::ToJson(::Json::Node& node) const
    {
        node.AddEnumeratedParameter("filterType", std::vector<std::string>({ "null", "box", "gaussian", "nlm" }), filterType);
        node.AddValue("radius", radius);
        node.AddValue("trigger", trigger);

        Json::Node nlmNode = node.AddChildObject("nlm");
        nlmNode.AddValue("alpha", nlm.alpha);
        nlmNode.AddValue("k", nlm.K);
    }

    __host__ void LightProbeKernelFilterParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetEnumeratedParameter("filterType", std::vector<std::string>({ "null", "box", "gaussian", "nlm" }), filterType, flags);
        node.GetValue("radius", radius, flags);
        node.GetValue("trigger", trigger, Json::kSilent);

        Json::Node nlmNode = node.GetChildObject("nlm", flags);
        if (nlmNode)
        {
            nlmNode.GetValue("alpha", nlm.alpha, flags);
            nlmNode.GetValue("k", nlm.K, flags);
        }
         
        radius = clamp(radius, 1e-3f, 20.0f);
        nlm.alpha = clamp(nlm.alpha, 0.0f, std::numeric_limits<float>::max());
        nlm.K = clamp(nlm.K, 0.0f, std::numeric_limits<float>::max());
    }
    
    __host__ Host::LightProbeKernelFilter::LightProbeKernelFilter(const ::Json::Node& node, const std::string& id) : 
        m_gridSize(1), m_blockSize(1)
    {
        FromJson(node, Json::kRequiredWarn);

        node.GetValue("inputGridID", m_inputGridID, Json::kRequiredAssert);
        node.GetValue("inputGridHalfID", m_inputGridHalfID, Json::kNotBlank);
        node.GetValue("outputGridID", m_outputGridID, Json::kRequiredAssert);
         
        AssertMsgFmt(!GlobalAssetRegistry::Get().Exists(m_outputGridID), "Error: an asset with ID '%s' already exists'.", m_outputGridID.c_str());

        // Create some objects
        m_hostOutputGrid = AssetHandle<Host::LightProbeGrid>(m_outputGridID, m_outputGridID);
        m_hostReduceBuffer = AssetHandle<Host::Array<vec3>>(new Host::Array<vec3>(m_hostStream), tfm::format("%s_reduceBuffer", id));
        m_hostReduceBuffer->Resize(1024 * 1024);
    }
    
    __host__ AssetHandle<Host::RenderObject> Host::LightProbeKernelFilter::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLightProbeFilter) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LightProbeKernelFilter(json, id), id);
    }

    __host__ void Host::LightProbeKernelFilter::FromJson(const ::Json::Node& node, const uint flags)
    {
        m_objects.params.FromJson(node, flags);

        Prepare();
    }

    __host__ void Host::LightProbeKernelFilter::OnDestroyAsset()
    {
        m_hostOutputGrid.DestroyAsset();
        m_hostReduceBuffer.DestroyAsset();
    }

    __host__ void Host::LightProbeKernelFilter::Bind(RenderObjectContainer& sceneObjects)
    {
        m_hostInputGrid = sceneObjects.FindByID(m_inputGridID).DynamicCast<Host::LightProbeGrid>();
        if (!m_hostInputGrid)
        {
            Log::Error("Error: LightProbeKernelFilter::Bind(): the specified input light probe grid '%s' is invalid.\n", m_inputGridID);
            return;
        }

        m_hostInputHalfGrid = nullptr;
        if (!m_inputGridHalfID.empty())
        {
            m_hostInputHalfGrid = sceneObjects.FindByID(m_inputGridHalfID).DynamicCast<Host::LightProbeGrid>();
            if (!m_hostInputHalfGrid)
            {
                Log::Error("Error: LightProbeKernelFilter::Bind(): the specified half input light probe grid '%s' is invalid.\n", m_inputGridHalfID);
                return;
            }
        }

        Prepare();   
    }

    __host__ void Host::LightProbeKernelFilter::Prepare()
    {
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        Assert(m_hostReduceBuffer);

        // Establish the dimensions of the kernel
        const auto& gridParams = m_hostInputGrid->GetParams();
        m_objects.gridDensity = gridParams.gridDensity;
        m_objects.numProbes = gridParams.numProbes;
        m_objects.coefficientsPerProbe = gridParams.coefficientsPerProbe;
        Assert(m_objects.coefficientsPerProbe <= kMaxCoefficients);      
      
        m_objects.kernelRadius = max(1, int(std::ceil(m_objects.params.radius)));

        const int maxVolume = m_hostReduceBuffer->Size() * kBlockSize / ((gridParams.coefficientsPerProbe + 1) * gridParams.numProbes);
        const int maxRadius = int((std::pow(float(maxVolume), 1.0f / 3.0f) - 1.0f) * 0.5f) - 1;
        if (m_objects.kernelRadius > maxRadius)
        {
            Log::Warning("Warning: filter kernel radius exceeds the capacity of the preset accumulation buffer. Constraining to %i.\n", maxRadius);
            m_objects.kernelRadius = maxRadius;
            m_objects.params.radius = maxRadius;
        }

        m_objects.kernelSpan = 2 * m_objects.kernelRadius + 1;
        m_objects.kernelVolume = cub(m_objects.kernelSpan);
        m_objects.blocksPerProbe = (m_objects.kernelVolume + (kBlockSize - 1)) / kBlockSize;

        m_probeRange = m_hostReduceBuffer->Size() / ((m_objects.coefficientsPerProbe + 1) * m_objects.blocksPerProbe);
        m_gridSize = min(m_objects.numProbes * m_objects.blocksPerProbe,
                         m_probeRange * (m_objects.coefficientsPerProbe + 1) * m_objects.blocksPerProbe);      

        Log::Debug("kernelVolume: %i\n", m_objects.kernelVolume);
        Log::Debug("blocksPerProbe: %i\n", m_objects.blocksPerProbe);
        Log::Debug("probeRange: %i\n", m_probeRange);
        Log::Debug("gridSize: %i\n", m_gridSize);

        // Initialise the output grid so it has the same dimensions as the input
        m_hostOutputGrid->Prepare(m_hostInputGrid->GetParams());

        m_objects.cu_inputGrid = m_hostInputGrid->GetDeviceInstance();
        m_objects.cu_inputHalfGrid = m_hostInputHalfGrid ? m_hostInputHalfGrid->GetDeviceInstance() : nullptr;
        m_objects.cu_outputGrid = m_hostOutputGrid->GetDeviceInstance();
        m_objects.cu_reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
    }

    __device__ float ComputeNLMWeight(Host::LightProbeKernelFilter::Objects& objects, const ivec3& pos0, const ivec3& posK)
    {        
        int numValidWeights = 0;
        float weights[kMaxCoefficients];
        memset(weights, 0, sizeof(vec3) * kMaxCoefficients);

        if (!objects.cu_inputHalfGrid) { return 0.0f; }
        
        // Iterate over the local block surrounding each element and compute the relative distance
        for (int w = -1; w <= 1; ++w)
        {
            for (int v = -1; v <= 1; ++v)
            {
                for (int u = -1; u <= 1; ++u)
                {
                    const int probeIdxM = objects.cu_inputGrid->IdxAt(pos0 + ivec3(u, v, w));
                    const int probeIdxN = objects.cu_inputGrid->IdxAt(pos0 + posK + ivec3(u, v, w));
                    if (probeIdxM < 0 || probeIdxN < 0) { continue; }

                    const vec3* probeM = objects.cu_inputGrid->At(probeIdxM);
                    const vec3* probeN = objects.cu_inputGrid->At(probeIdxN);

                    if (probeM[objects.coefficientsPerProbe - 1].x < 0.5f || probeN[objects.coefficientsPerProbe - 1].x < 0.5f) { continue; }

                    const vec3* probeHalfM = objects.cu_inputHalfGrid->At(probeIdxM);
                    const vec3* probeHalfN = objects.cu_inputHalfGrid->At(probeIdxN);
                    
                    for (int coeffIdx = 0; coeffIdx < objects.coefficientsPerProbe - 1; ++coeffIdx)
                    {                          
                        const vec3& M = probeM[coeffIdx];
                        const vec3& N = probeN[coeffIdx];
                        const vec3& halfM = probeHalfM[coeffIdx];
                        const vec3& halfN = probeHalfN[coeffIdx];
                        const vec3 varN = sqr((N - halfN) - halfN) * 2.0f;
                        const vec3 varM = sqr((M - halfM) - halfM) * 2.0f;
                        
                        const vec3 d2 = (sqr(M - N) - objects.params.nlm.alpha * (varN + varM)) /
                                        (vec3(1e-10f) + sqr(objects.params.nlm.K) * (varN + varM));
                        weights[coeffIdx] += expf(-max(0.0f, cwiseMax(d2)));
                    }
                    numValidWeights++;
                }
            }
        }

        // Find the coefficient with the maximum weight and return
        float maxWeight = kFltMax;
        for (int coeffIdx = 0; coeffIdx < objects.coefficientsPerProbe - 1; ++coeffIdx)
        {
            maxWeight = min(maxWeight, weights[coeffIdx]);
        }
        return maxWeight / float(max(1, numValidWeights));
    }

    __global__ void KernelFilterGaussian(Host::LightProbeKernelFilter::Objects objects, const int probeStartIdx)
    {
        assert(objects.cu_inputGrid);
        assert(objects.cu_outputGrid);
        
        __shared__ vec3 weightedCoeffs[kBlockSize * kMaxCoefficients];
        __shared__ float weights[kBlockSize];
        __shared__ int coefficientsPerProbe;
        __shared__ int blocksPerProbe;
        __shared__ int kernelSpan;
        __shared__ ivec3 gridDensity;
        //__shared__ float kernelWeights[21];

        assert(objects.params.radius <= 20);

        // Copy some common data into shared memory
        coefficientsPerProbe = objects.coefficientsPerProbe;
        blocksPerProbe = objects.blocksPerProbe;
        kernelSpan = objects.kernelSpan;
        gridDensity = objects.gridDensity;
        memset(weightedCoeffs, 0, sizeof(vec3) * kBlockSize * kMaxCoefficients);

        // Precompute the Gaussian kernel
        /*if (kKernelIdx == 0 && objects.params.filterType == kKernelFilterGaussian)
        {
            for (int i = 0; i < 21; ++i)
            {
                kernelWeights[i] = Integrate1DGaussian(i - 0.5f, i + 0.5f, objects.params.radius);
            }
        }*/

        __syncthreads();

        // Get the index of the probe in the grid and the sample in the kernel
        const int probeIdx0 = probeStartIdx + blockIdx.x / objects.blocksPerProbe;
        const int probeIdxK = kBlockSize * (blockIdx.x % objects.blocksPerProbe) + threadIdx.x;

        assert(probeIdx0 < objects.numProbes);

        // If the index of the element lies outside the kernel, zero its weight
        if (probeIdxK >= objects.kernelVolume)
        {
            weights[threadIdx.x] = 0.0f;
        }
        else
        {
            // Compute the sample position relative to the centre of the kernel
            const ivec3 posK = ivec3(probeIdxK % objects.kernelSpan,
                                          (probeIdxK / objects.kernelSpan) % objects.kernelSpan,
                                           probeIdxK / (objects.kernelSpan * objects.kernelSpan)) - ivec3(objects.kernelRadius);

            // Compute the absolute position at the origin of the kernel
            const ivec3 pos0 = ivec3(probeIdx0 % objects.gridDensity.x,
                                          (probeIdx0 / objects.gridDensity.x) % objects.gridDensity.y,
                                           probeIdx0 / (objects.gridDensity.x * objects.gridDensity.y));

            // If the neighbourhood probe lies outside the bounds of the grid, set the weight to zero
            const int probeIdxN = objects.cu_inputGrid->IdxAt(pos0 + posK);
            if (probeIdxN < 0)
            {
                weights[threadIdx.x] = 0.0f;
            }
            else
            {
                float weight = 1.0f;
                if (posK != ivec3(0))
                {
                    // Calculate the weight for the sample
                    switch (objects.params.filterType)
                    {
                    case kKernelFilterGaussian:
                    {
                        const float len = length(vec3(posK));
                        if (len <= objects.params.radius)
                        {
                            weight = Integrate1DGaussian(len - 0.5f, len + 0.5f, objects.params.radius);
                        }
                        break;
                    }
                    case kKernelFilterNLM:
                        weight = ComputeNLMWeight(objects, pos0, posK);
                        break;
                    };
                }

                weights[threadIdx.x] = weight;
                if (weight > 0.0f)
                {
                    // Accumulate the weighted coefficients
                    const vec3* inputCoeff = objects.cu_inputGrid->At(probeIdxN);
                    for (int coeffIdx = 0; coeffIdx < coefficientsPerProbe - 1; ++coeffIdx)
                    {
                        weightedCoeffs[threadIdx.x * kMaxCoefficients + coeffIdx] = inputCoeff[coeffIdx] * weight;
                    }
                }
            }
        }

        __syncthreads();

        // Reduce the contents of the shared bufffer
        for (int interval = kBlockSize >> 1; interval > 0; interval >>= 1)
        {
            if (threadIdx.x < interval && weights[threadIdx.x + interval] > 0.0f)
            {
                for (int coeffIdx = 0; coeffIdx < coefficientsPerProbe - 1; ++coeffIdx)
                {
                    weightedCoeffs[threadIdx.x * kMaxCoefficients + coeffIdx] += weightedCoeffs[(threadIdx.x + interval) * kMaxCoefficients + coeffIdx];
                }
                weights[threadIdx.x] += weights[threadIdx.x + interval];
            }

            __syncthreads();
        }
        
        if (threadIdx.x == 0)
        {
            // If the entire convolution operation fits into a single block, copy straight into the output buffer
            if (objects.blocksPerProbe == 1)
            {
                vec3* outputBuffer = objects.cu_outputGrid->At(probeIdx0);
                for (int coeffIdx = 0; coeffIdx < coefficientsPerProbe; ++coeffIdx)
                {
                    outputBuffer[coeffIdx] = weightedCoeffs[coeffIdx] / weights[0];
                }
            }
            else
            {
                // Copy the data into the intermediate buffer. 
                // We store the weighted SH coefficients as vec3s followed by a final vec3 whose first element contains the weight.
                assert(objects.cu_reduceBuffer);
                const int bufferIdx = (probeIdx0 * objects.blocksPerProbe + (probeIdxK / kBlockSize)) * (objects.coefficientsPerProbe + 1);
                assert(bufferIdx < objects.cu_reduceBuffer->Size());
                vec3* outputBuffer = &(*objects.cu_reduceBuffer)[bufferIdx];
                for (int coeffIdx = 0; coeffIdx < coefficientsPerProbe; ++coeffIdx)
                {
                    outputBuffer[coeffIdx] = weightedCoeffs[coeffIdx];
                }
                outputBuffer[coefficientsPerProbe].x = weights[0];
            }
        }
    } 

    __global__ void KernelCopyFromReduceBuffer(Host::LightProbeKernelFilter::Objects objects)
    {       
        assert(objects.cu_inputGrid);
        assert(objects.cu_outputGrid);
        
        if (kKernelX >= objects.numProbes) { return; }
        
        vec3* outputBuffer = objects.cu_outputGrid->At(kKernelX);
        memset(outputBuffer, 0, sizeof(vec3) * objects.coefficientsPerProbe);
     
        const vec3* reduceCoeff = &(*objects.cu_reduceBuffer)[kKernelX * objects.blocksPerProbe * (objects.coefficientsPerProbe + 1)];
        
        // Sum the coefficients and weights over all the blocks
        float sumWeights = 0.0;
        for (int blockIdx = 0, reduceIdx = 0; blockIdx < objects.blocksPerProbe; ++blockIdx)
        {
            for (int coeffIdx = 0; coeffIdx < objects.coefficientsPerProbe; ++coeffIdx, ++reduceIdx)
            {                
                outputBuffer[coeffIdx] += reduceCoeff[reduceIdx];
            }
            sumWeights += reduceCoeff[reduceIdx++].x;
        }

        // Normalise the accumulated coefficients by the sum of the kernel weights
        for (int coeffIdx = 0; coeffIdx < objects.coefficientsPerProbe; ++coeffIdx)
        {
            outputBuffer[coeffIdx] /= sumWeights;
        }
    }

    __host__ void Host::LightProbeKernelFilter::OnPostRenderPass()
    {              
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }
        
        // Pass-through filter just copies the data
        if (m_objects.params.filterType == kKernelFilterNull)
        {
            m_hostOutputGrid->Replace(*m_hostInputGrid);
            return;
        }
        
        for (int probeIdx = 0; probeIdx < m_objects.numProbes; probeIdx += m_probeRange)
        {
            KernelFilterGaussian << <m_gridSize, kBlockSize, 0, m_hostStream >> > (m_objects, probeIdx);
            
            if (m_objects.blocksPerProbe > 1)
            {
                KernelCopyFromReduceBuffer << < (m_objects.numProbes + 255) / 256, 256, 0, m_hostStream >> > (m_objects);
            }

            IsOk(cudaStreamSynchronize(m_hostStream));
        }
    }

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::LightProbeKernelFilter::GetChildObjectHandles()
    {
        std::vector<AssetHandle<Host::RenderObject>> objects;
        objects.emplace_back(m_hostOutputGrid);
        return objects;
    }

    __host__ void Host::LightProbeKernelFilter::OnUpdateSceneGraph(RenderObjectContainer& sceneObjects)
    {
        if (m_hostInputGrid && m_hostOutputGrid &&
            m_hostInputGrid->GetParams() != m_hostOutputGrid->GetParams())
        {
            Prepare();
        }
    }
}