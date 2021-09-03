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
        m_hostReduceBuffer = AssetHandle<Host::Array<float>>(new Host::Array<float>(m_hostStream), tfm::format("%s_reduceBuffer", id));
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

        // Prepare() will be called with OnUpdateSceneGraph()
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
    }

    __global__ void KernelFilterGaussian(const Host::LightProbeKernelFilter::Objects objects, const int probeStartIdx)
    {
        __shared__ vec3 coeffData[kBlockSize * kMaxCoefficients];
        
        assert(objects.cu_inputGrid);
        assert(objects.cu_outputGrid); 

        // Get the index of the probe in the grid and the sample in the kernel
        const int probeIdx = probeStartIdx + blockIdx.x / objects.blocksPerProbe;
        const int sampleIdx = kBlockSize * (blockIdx.x % objects.blocksPerProbe) + threadIdx.x;

        if (sampleIdx >= objects.kernelVolume) { return; }

        // Compute the sample position relative to the centre of 
        const ivec3 samplePos = ivec3(sampleIdx % objects.kernelSpan,
                                      (sampleIdx / objects.kernelSpan) % objects.kernelSpan,
                                       sampleIdx / (objects.kernelSpan * objects.kernelSpan)) - ivec3(objects.kernelRadius);

        const ivec3 probePos = ivec3(probeIdx % objects.gridDensity.x,
                                      (probeIdx / objects.gridDensity.x) % objects.gridDensity.y,
                                       probeIdx / (objects.gridDensity.x * objects.gridDensity.y));

        if (sampleIdx == 0)
        {
            const int probeIdx = objects.cu_inputGrid->IdxAt(probePos);
            assert(probeIdx >= 0);

            const vec3* inputCoeff = objects.cu_inputGrid->At(probeIdx);
            vec3* outputCoeff = objects.cu_outputGrid->At(probeIdx);
            for (int coeffIdx = 0; coeffIdx < objects.coefficientsPerProbe - 1; ++coeffIdx)
            {
                outputCoeff[coeffIdx] = inputCoeff[coeffIdx];
            }
        }
    } 

    __host__ void Host::LightProbeKernelFilter::OnPostRenderPass()
    {              
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }
        
        for (int probeIdx = 0; probeIdx < m_objects.numProbes; probeIdx += m_probeRange)
        {
            KernelFilterGaussian << <m_gridSize, kBlockSize, 0, m_hostStream >> > (m_objects, probeIdx);
        }

        IsOk(cudaStreamSynchronize(m_hostStream));
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