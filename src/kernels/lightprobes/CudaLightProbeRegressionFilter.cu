#include "CudaLightProbeRegressionFilter.cuh"
#include "../CudaManagedArray.cuh"

#include "generic/JsonUtils.h"

#define kBlockSize 256
#define kMaxCoefficients 5

namespace Cuda
{
    __host__ __device__ LightProbeRegressionFilterParams::LightProbeRegressionFilterParams() :
        polynomialOrder(0),
        radius(1)
    {

    }

    __host__ void LightProbeRegressionFilterParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("polynomialOrder", polynomialOrder);
        node.AddValue("radius", radius);
    
    }

    __host__ void LightProbeRegressionFilterParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("polynomialOrder", polynomialOrder, flags);
        node.GetValue("radius", radius, flags);

        radius = clamp(radius, 1, 10);
    }

    __host__ Host::LightProbeRegressionFilter::LightProbeRegressionFilter(const ::Json::Node& node, const std::string& id) :
        m_gridSize(1), m_blockSize(1)
    {
        FromJson(node, Json::kRequiredWarn);

        node.GetValue("inputGridID", m_inputGridID, Json::kRequiredAssert);
        node.GetValue("inputGridHalfID", m_inputGridHalfID, Json::kNotBlank);
        node.GetValue("outputGridID", m_outputGridID, Json::kRequiredAssert);

        AssertMsgFmt(!GlobalAssetRegistry::Get().Exists(m_outputGridID), "Error: an asset with ID '%s' already exists'.", m_outputGridID.c_str());

        // Create some objects
        m_hostOutputGrid = AssetHandle<Host::LightProbeGrid>(m_outputGridID, m_outputGridID);
        m_hostPolyCoeffs = AssetHandle<Host::Array<vec3>>(new Host::Array<vec3>(m_hostStream), tfm::format("%s_polyCoeffs", id));
        m_hostRegressionWeights = AssetHandle<Host::Array<float>>(new Host::Array<float>(m_hostStream), tfm::format("%s_regressionWeights", id));
    }

    __host__ AssetHandle<Host::RenderObject> Host::LightProbeRegressionFilter::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLightProbeFilter) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LightProbeRegressionFilter(json, id), id);
    }

    __host__ void Host::LightProbeRegressionFilter::FromJson(const ::Json::Node& node, const uint flags)
    {
        m_objects->params.FromJson(node, flags);

        Prepare();
    }

    __host__ void Host::LightProbeRegressionFilter::OnDestroyAsset()
    {
        m_hostOutputGrid.DestroyAsset();
        m_hostPolyCoeffs.DestroyAsset();
        m_hostRegressionWeights.DestroyAsset();
    }

    __host__ void Host::LightProbeRegressionFilter::Bind(RenderObjectContainer& sceneObjects)
    {
        m_hostInputGrid = sceneObjects.FindByID(m_inputGridID).DynamicCast<Host::LightProbeGrid>();
        if (!m_hostInputGrid)
        {
            Log::Error("Error: LightProbeRegressionFilter::Bind(): the specified input light probe grid '%s' is invalid.\n", m_inputGridID);
            return;
        }

        m_hostInputHalfGrid = nullptr;
        if (!m_inputGridHalfID.empty())
        {
            m_hostInputHalfGrid = sceneObjects.FindByID(m_inputGridHalfID).DynamicCast<Host::LightProbeGrid>();
            if (!m_hostInputHalfGrid)
            {
                Log::Error("Error: LightProbeRegressionFilter::Bind(): the specified half input light probe grid '%s' is invalid.\n", m_inputGridHalfID);
                return;
            }
        }

        Prepare();
    }

    __host__ void Host::LightProbeRegressionFilter::Prepare()
    {
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        Assert(m_hostPolyCoeffs);

        // Establish the dimensions of the kernel
        const auto& gridParams = m_hostInputGrid->GetParams();
        m_objects->gridData.density = gridParams.gridDensity;
        m_objects->gridData.numProbes = gridParams.numProbes;
        m_objects->gridData.coefficientsPerProbe = gridParams.coefficientsPerProbe;
        Assert(m_objects->gridData.coefficientsPerProbe <= kMaxCoefficients);

        /*const int newSize = m_params.numProbes * m_params.coefficientsPerProbe;
        int arraySize = m_shData->Size();

        // Resize the array as a power of two 
        if (arraySize < newSize)
        {
            for (arraySize = 1; arraySize < newSize; arraySize <<= 1) {}

            Log::Debug("Resized to %i\n", arraySize);
            m_shData->Resize(arraySize);
            m_validityData->Resize(arraySize);
        }*/

        // Initialise the output grid so it has the same dimensions as the input
        m_hostOutputGrid->Prepare(m_hostInputGrid->GetParams());

        m_objects->gridData.cu_inputGrid = m_hostInputGrid->GetDeviceInstance();
        m_objects->gridData.cu_inputHalfGrid = m_hostInputHalfGrid ? m_hostInputHalfGrid->GetDeviceInstance() : nullptr;
        m_objects->gridData.cu_outputGrid = m_hostOutputGrid->GetDeviceInstance();
        m_objects->cu_polyCoeffs = m_hostPolyCoeffs->GetDeviceInstance();
        m_objects->cu_regressionWeights = m_hostRegressionWeights->GetDeviceInstance();
        m_objects.Upload();
    }

    __global__ void KernelRandomisePolynomialCoefficients(Host::LightProbeRegressionFilter::Objects* objects)
    {
    }

    __global__ void KernelComputeRegressionWeights(Host::LightProbeRegressionFilter::Objects* objects, const int probeStartIdx)
    {

    }

    __global__ void KernelApplyRegressionIteration(Host::LightProbeRegressionFilter::Objects* objects)
    {
    }

    __host__ void Host::LightProbeRegressionFilter::OnPostRenderPass()
    {
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }
        
        //KernelApplyRegressionStep << <1, 1, 1 >> > (m_objects.GetDeviceObject());
    }

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::LightProbeRegressionFilter::GetChildObjectHandles()
    {
        std::vector<AssetHandle<Host::RenderObject>> objects;
        objects.emplace_back(m_hostOutputGrid);
        return objects;
    }

    __host__ void Host::LightProbeRegressionFilter::OnUpdateSceneGraph(RenderObjectContainer& sceneObjects)
    {
        if (m_hostInputGrid && m_hostOutputGrid &&
            m_hostInputGrid->GetParams() != m_hostOutputGrid->GetParams())
        {
            Prepare();
        }
    }
}