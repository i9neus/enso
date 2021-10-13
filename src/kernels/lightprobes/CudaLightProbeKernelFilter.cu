#include "CudaLightProbeKernelFilter.cuh"
#include "../CudaManagedArray.cuh"

#include "generic/JsonUtils.h"

#define kBlockSize 256
#define kMaxCoefficients 5

namespace Cuda
{
    __host__ __device__ LightProbeKernelFilterParams::LightProbeKernelFilterParams() : 
        filterType(kKernelFilterGaussian),
        kernelRadius(1)
    {
    }
    
    __host__ void LightProbeKernelFilterParams::ToJson(::Json::Node& node) const
    {
        node.AddEnumeratedParameter("filterType", std::vector<std::string>({ "null", "box", "gaussian", "nlm", "nlmconst" }), filterType);
        node.AddValue("radius", kernelRadius);

        Json::Node nlmNode = node.AddChildObject("nlm");
        nlm.ToJson(nlmNode);
    }

    __host__ void LightProbeKernelFilterParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetEnumeratedParameter("filterType", std::vector<std::string>({ "null", "box", "gaussian", "nlm", "nlmconst" }), filterType, flags);
        node.GetValue("radius", kernelRadius, flags);

        Json::Node nlmNode = node.GetChildObject("nlm", flags);
        if (nlmNode) { nlm.FromJson(nlmNode, flags); }
         
        kernelRadius = clamp(kernelRadius, 0, 10);
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
        
        // FIXME: Static size; needs to be dynamic. 
        m_hostReduceBuffer->Resize(1024 * 1024);
    }
    
    __host__ AssetHandle<Host::RenderObject> Host::LightProbeKernelFilter::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLightProbeFilter) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LightProbeKernelFilter(json, id), id);
    }

    __host__ void Host::LightProbeKernelFilter::FromJson(const ::Json::Node& node, const uint flags)
    {
        m_objects->params.FromJson(node, flags);

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
        m_isActive = true;
        
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        Assert(m_hostReduceBuffer);

        // Establish the dimensions of the kernel   
        auto& gridData = m_objects->gridData.Initialise(m_hostInputGrid, m_hostInputGrid, m_hostInputHalfGrid, m_hostOutputGrid);

        Assert(gridData.coefficientsPerProbe <= kMaxCoefficients);

        m_objects->kernelRadius = m_objects->params.kernelRadius;

        const auto& gridParams = m_hostInputGrid->GetParams();

        m_objects->kernelSpan = 2 * m_objects->kernelRadius + 1;
        m_objects->kernelVolume = cub(m_objects->kernelSpan);
        m_objects->blocksPerProbe = (m_objects->kernelVolume + (kBlockSize - 1)) / kBlockSize;

        /*if (m_objects->blocksPerProbe == 1)
        {
            m_probeRange = gridData.numProbes;
            m_gridSize = gridData.numProbes;
        }
        else*/
        {
            m_probeRange = min(uint(gridData.numProbes),
                               m_hostReduceBuffer->Size() / ((gridData.coefficientsPerProbe + 1) * m_objects->blocksPerProbe));
            m_gridSize = m_probeRange * m_objects->blocksPerProbe;
        }
      
        Log::Debug("Grid dimensions: %s\n", gridParams.gridDensity.format());
        Log::Debug("kernelVolume: %i\n", m_objects->kernelVolume);
        Log::Debug("blocksPerProbe: %i\n", m_objects->blocksPerProbe);
        Log::Debug("probeRange: %i\n", m_probeRange);
        Log::Debug("gridSize: %i\n", m_gridSize);

        // Initialise the output grid so it has the same dimensions as the input
        m_hostOutputGrid->Prepare(gridParams);
        
        m_objects->cu_reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
        m_objects.Upload();
    }

    __global__ void KernelFilter(Host::LightProbeKernelFilter::Objects* const objectsPtr, const int probeStartIdx)
    {
        __shared__ Host::LightProbeKernelFilter::Objects objects;
        __shared__ vec3 weightedCoeffs[kBlockSize * kMaxCoefficients];
        __shared__ float weights[kBlockSize];

        if (kThreadIdx == 0)
        {
            assert(objectsPtr);
            objects = *objectsPtr;
            assert(objects.gridData.cu_inputGrid);
            assert(objects.gridData.cu_outputGrid);
            assert(objects.params.kernelRadius <= 20);

            memset(weightedCoeffs, 0, sizeof(vec3) * kBlockSize * kMaxCoefficients);
        }    

        __syncthreads();

        // Get the index of the probe in the grid and the sample in the kernel
        const int probeIdx0 = probeStartIdx + blockIdx.x / objects.blocksPerProbe;
        if (probeIdx0 >= objects.gridData.numProbes) { return; }

        auto& gridData = objects.gridData;
        const auto& params = objects.params;
        const int probeIdxK = kBlockSize * (blockIdx.x % objects.blocksPerProbe) + threadIdx.x;

        // If the index of the element lies outside the kernel, zero its weight
        if (probeIdxK >= objects.kernelVolume)
        {
            weights[threadIdx.x] = 0.0f;
        }
        else
        {
            // Compute the sample position relative to the centre of the kernel
            const ivec3 posK = GridPosFromProbeIdx(probeIdxK, objects.kernelSpan) - ivec3(objects.kernelRadius);

            // Compute the absolute position at the origin of the kernel
            const ivec3 pos0 = GridPosFromProbeIdx(probeIdx0, objects.gridData.density);

            // If the neighbourhood probe lies outside the bounds of the grid, set the weight to zero
            const int probeIdxN = gridData.cu_inputGrid->IdxAt(pos0 + posK);
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
                    switch (params.filterType)
                    {
                    case kKernelFilterGaussian:
                    {
                        const float len = length(vec3(posK));
                        if (len >= params.kernelRadius + 1.0f) { weight = 0.0f; }
                        else
                        {
                            //weight = Integrate1DGaussian(len - 0.5f, len + 0.5f, params.kernelRadius);
                            weight = 1.0f - sqr(len / float(params.kernelRadius + 1.0f));
                        }       
                        break;
                    }
                    case kKernelFilterNLM:
                        weight = ComputeNLMWeight(gridData, params.nlm, pos0, posK);
                        break;
                    case kKernelFilterNLMConst:
                        weight = ComputeConstVarianceNLMWeight(gridData, params.nlm, pos0, posK);
                        break;
                    };
                }

                weights[threadIdx.x] = weight;
                if (weight > 0.0f)
                {
                    // Accumulate the weighted coefficients
                    const vec3* inputCoeff = gridData.cu_inputGrid->At(probeIdxN);
                    for (int coeffIdx = 0; coeffIdx < gridData.shCoeffsPerProbe; ++coeffIdx)
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
                for (int coeffIdx = 0; coeffIdx < gridData.shCoeffsPerProbe; ++coeffIdx)
                {
                    weightedCoeffs[threadIdx.x * kMaxCoefficients + coeffIdx] += weightedCoeffs[(threadIdx.x + interval) * kMaxCoefficients + coeffIdx];
                }
                weights[threadIdx.x] += weights[threadIdx.x + interval];
            }

            __syncthreads();
        }
        
        if (kThreadIdx == 0)
        {           
            // If the entire convolution operation fits into a single block, copy straight into the output buffer
            /*if (objects.blocksPerProbe == 1)
            {
                vec3* outputBuffer = gridData.cu_outputGrid->At(probeIdx0);
                for (int coeffIdx = 0; coeffIdx < gridData.shCoeffsPerProbe; ++coeffIdx)
                {
                    outputBuffer[coeffIdx] = weightedCoeffs[coeffIdx] / weights[0];
                }
                // Don't filter the metadata. Just copy.
                outputBuffer[gridData.shCoeffsPerProbe] = gridData.cu_inputGrid->At(probeIdx0)[gridData.shCoeffsPerProbe];
            }
            else*/
            {
                // Copy the data into the intermediate buffer. 
                // We store the weighted SH coefficients as vec3s followed by a final vec3 whose first element contains the weight.
                assert(objects.cu_reduceBuffer);
                const int outputIdx = blockIdx.x * (gridData.coefficientsPerProbe + 1);
                assert(outputIdx < objects.cu_reduceBuffer->Size());
                vec3* outputBuffer = &(*objects.cu_reduceBuffer)[outputIdx];
                for (int coeffIdx = 0; coeffIdx < gridData.coefficientsPerProbe; ++coeffIdx)
                {
                    outputBuffer[coeffIdx] = weightedCoeffs[coeffIdx];
                }
                outputBuffer[gridData.coefficientsPerProbe].x = weights[0];
            }
        }
    } 

    __global__ void KernelCopyFromReduceBuffer(Host::LightProbeKernelFilter::Objects* objects, const int probeStartIdx)
    {       
        assert(objects->gridData.cu_inputGrid);
        assert(objects->gridData.cu_outputGrid);
        
        if (probeStartIdx + kKernelX >= objects->gridData.numProbes) { return; }
        
        vec3* outputBuffer = objects->gridData.cu_outputGrid->At(probeStartIdx + kKernelX);
        memset(outputBuffer, 0, sizeof(vec3) * objects->gridData.coefficientsPerProbe);
     
        const vec3* reduceCoeff = &(*objects->cu_reduceBuffer)[kKernelX * objects->blocksPerProbe * (objects->gridData.coefficientsPerProbe + 1)];
        
        // Sum the coefficients and weights over all the blocks
        float sumWeights = 0.0;
        for (int blockIdx = 0, reduceIdx = 0; blockIdx < objects->blocksPerProbe; ++blockIdx)
        {
            for (int coeffIdx = 0; coeffIdx < objects->gridData.shCoeffsPerProbe; ++coeffIdx, ++reduceIdx)
            {                                
                outputBuffer[coeffIdx] += reduceCoeff[reduceIdx];
            }
            reduceIdx++;
            sumWeights += reduceCoeff[reduceIdx++].x;
        }

        // Normalise the accumulated coefficients by the sum of the kernel weights
        for (int coeffIdx = 0; coeffIdx < objects->gridData.shCoeffsPerProbe; ++coeffIdx)
        {
            outputBuffer[coeffIdx] /= sumWeights;
        }
    }

    __host__ void Host::LightProbeKernelFilter::OnPostRenderPass()
    {              
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }        
        
        // Pass-through filter just copies the data
        if (m_objects->params.filterType == kKernelFilterNull || !m_hostInputGrid->IsConverged())
        {
            m_hostOutputGrid->Replace(*m_hostInputGrid);
            m_isActive = true;
            return;
        }

        // If the input grid is requesting a filter op, enable the filter
        if (m_hostInputGrid->GetSemaphore("tag_do_filter") == true)
        {
            m_hostInputGrid->SetSemaphore("tag_do_filter", false);
            m_isActive = true;
        }

        if (!m_isActive) { return; }
        
        for (int probeIdx = 0; probeIdx < m_objects->gridData.numProbes; probeIdx += m_probeRange)
        {
            KernelFilter << <m_gridSize, kBlockSize, 0, m_hostStream >> > (m_objects.GetDeviceObject(), probeIdx);
            
            //if (m_objects->blocksPerProbe > 1)
            {
                KernelCopyFromReduceBuffer << < (m_probeRange + 255) / 256, 256, 0, m_hostStream >> > (m_objects.GetDeviceObject(), probeIdx);
            }

            IsOk(cudaStreamSynchronize(m_hostStream));
        }

        m_hostOutputGrid->SetSemaphore("tag_is_filtered", true);
        m_isActive = false;
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