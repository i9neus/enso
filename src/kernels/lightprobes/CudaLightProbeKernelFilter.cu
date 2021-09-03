#include "CudaLightProbeKernelFilter.cuh"
#include "../CudaManagedArray.cuh"
#include "CudaFilters.cuh"

#include "generic/JsonUtils.h"

#define kBlockSize 256
#define kMaxCoefficients 5

namespace Cuda
{
    __host__ void LightProbeKernelFilterParams::ToJson(::Json::Node& node) const
    {
        node.AddEnumeratedParameter("filterType", std::vector<std::string>({ "gaussian" }), filterType);
        node.AddValue("radius", radius);
        node.AddValue("trigger", trigger);
    }

    __host__ void LightProbeKernelFilterParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetEnumeratedParameter("filterType", std::vector<std::string>({ "gaussian" }), filterType, flags);
        node.GetValue("radius", radius, flags);
        node.GetValue("trigger", trigger, Json::kSilent);
    }
    
    __host__ Host::LightProbeKernelFilter::LightProbeKernelFilter(const ::Json::Node& node, const std::string& id) : 
        m_gridSize(1), m_blockSize(1)
    {
        FromJson(node, Json::kRequiredWarn);

        node.GetValue("inputGridID", m_inputGridID, Json::kRequiredAssert);
        node.GetValue("outputGridID", m_outputGridID, Json::kRequiredAssert);
         
        AssertMsgFmt(!GlobalAssetRegistry::Get().Exists(m_outputGridID), "Error: an asset with ID '%s' already exists'.", m_outputGridID.c_str());

        // Create some objects
        m_hostOutputGrid = AssetHandle<Host::LightProbeGrid>(m_outputGridID, m_outputGridID);
        m_hostReduceBuffer = AssetHandle<Host::Array<vec3>>(new Host::Array<vec3>(m_hostStream), tfm::format("%s_reduceBuffer", id));
        m_hostReduceBuffer->Resize(512 * 512);
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

        Prepare();   
    }

    __host__ void Host::LightProbeKernelFilter::Prepare()
    {
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }
        
        // Establish the dimensions of the kernel
        const auto& gridParams = m_hostInputGrid->GetParams();
        m_objects.gridDensity = gridParams.gridDensity;
        m_objects.numProbes = gridParams.numProbes;
        m_objects.coefficientsPerProbe = gridParams.coefficientsPerProbe;
        Assert(m_objects.coefficientsPerProbe <= kMaxCoefficients);
        m_objects.kernelRadius = max(1, int(std::ceil(m_objects.params.radius)));
        m_objects.kernelSpan = 2 * m_objects.kernelRadius + 1;
        m_objects.kernelVolume = cub(m_objects.kernelSpan);
        m_objects.blocksPerProbe = (m_objects.kernelVolume + (kBlockSize - 1)) / kBlockSize;        

        // If the volume of the kernel is smaller than the block size, there's no need to do an intermediate accumulation step
        /*if (m_objects.blocksPerProbe == 1)
        {
            m_probeRange = m_objects.numProbes;
            m_gridSize = m_objects.numProbes;
        }
        else*/
        {
            m_probeRange = m_hostReduceBuffer->Size() / ((m_objects.coefficientsPerProbe + 1) * m_objects.blocksPerProbe);
            m_gridSize = min(m_objects.numProbes * m_objects.blocksPerProbe,
                             m_probeRange * (m_objects.coefficientsPerProbe + 1) * m_objects.blocksPerProbe);
        } 

        Log::Debug("kernelVolume: %i\n", m_objects.kernelVolume);
        Log::Debug("blocksPerProbe: %i\n", m_objects.blocksPerProbe);
        Log::Debug("probeRange: %i\n", m_probeRange);
        Log::Debug("gridSize: %i\n", m_gridSize);

        // Initialise the output grid so it has the same dimensions as the input
        m_hostOutputGrid->Prepare(m_hostInputGrid->GetParams());

        m_objects.cu_inputGrid = m_hostInputGrid->GetDeviceInstance();
        m_objects.cu_outputGrid = m_hostOutputGrid->GetDeviceInstance();
        m_objects.cu_reduceBuffer = m_hostReduceBuffer->GetDeviceInstance();
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

        // Copy some common data into shared memory
        coefficientsPerProbe = objects.coefficientsPerProbe;
        blocksPerProbe = objects.blocksPerProbe;
        kernelSpan = objects.kernelSpan;
        gridDensity = objects.gridDensity;
        memset(weightedCoeffs, 0, sizeof(vec3) * kBlockSize * kMaxCoefficients);

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
                // Calculate the weight for the sample
                const float weight = 1.0f;
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
                vec3* outputBuffer = &(*objects.cu_reduceBuffer)[blockIdx.x * (objects.coefficientsPerProbe + 1)];
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