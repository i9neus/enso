#include "CudaLightProbeRegressionFilter.cuh"
#include "../CudaManagedArray.cuh"
#include "../CudaSampler.cuh"

#include "generic/JsonUtils.h"

#define kBlockSize 256
#define kMaxCoefficients 5

namespace Cuda
{
    __host__ __device__ LightProbeRegressionFilterParams::LightProbeRegressionFilterParams() :
        polynomialOrder(0),
        regressionRadius(1),
        reconstructionRadius(1),
        isNullFilter(true)
    {

    }

    __host__ void LightProbeRegressionFilterParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("polynomialOrder", polynomialOrder);
        node.AddValue("regressionRadius", regressionRadius);
        node.AddValue("reconstructionRadius", reconstructionRadius);
        node.AddValue("isNullFilter", isNullFilter);
    }

    __host__ void LightProbeRegressionFilterParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("polynomialOrder", polynomialOrder, flags);
        node.GetValue("regressionRadius", regressionRadius, flags);
        node.GetValue("reconstructionRadius", reconstructionRadius, flags);
        node.GetValue("isNullFilter", isNullFilter, flags);

        regressionRadius = clamp(regressionRadius, 0, 10);
        reconstructionRadius = clamp(reconstructionRadius, 0, 10);
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
        m_hostRegressionWeights->Resize(1024 * 1024);
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

    __global__ void KernelRandomisePolynomialCoefficients(Device::Array<vec3>* coeffBuffer)
    {
        assert(coeffBuffer);
        assert(kKernelIdx < coeffBuffer->Size());

        PseudoRNG rng(HashOf(uint(kKernelIdx)));
        (*coeffBuffer)[kKernelIdx] = rng.Rand<0, 1, 2>();
    }

    __host__ void Host::LightProbeRegressionFilter::Prepare()
    {
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        Assert(m_hostPolyCoeffs);

        // Establish the dimensions of the kernel
        auto& gridData = m_objects->gridData.Prepare(m_hostInputGrid, m_hostInputHalfGrid, m_hostOutputGrid);
        Assert(m_objects->gridData.coefficientsPerProbe <= kMaxCoefficients);

        // Each coefficient stores the coefficients of the fitted polynomial plus 2 additional coefficients which represent the max and min values of the kernel
        m_objects->polyCoeffsPerCoefficient = cub(m_objects->params.polynomialOrder + 1) + 2;
        m_objects->polyCoeffsPerProbe = m_objects->polyCoeffsPerCoefficient * gridData.coefficientsPerProbe;
        m_objects->numPolyCoeffs = m_objects->polyCoeffsPerProbe * gridData.numProbes;
        m_objects->reconstructionKernelVolume = cub(m_objects->params.reconstructionRadius * 2 + 1);

        // Resize the polynomial coefficient array as a power of two 
        if(m_hostPolyCoeffs->ExpandToNearestPow2(m_objects->numPolyCoeffs))
        {
            Log::Debug("Resized m_hostPolyCoeffs to %i\n", m_hostPolyCoeffs->Size());
        }

        // Initialise the output grid so it has the same dimensions as the input
        m_hostOutputGrid->Prepare(m_hostInputGrid->GetParams());
                
        m_objects->cu_polyCoeffs = m_hostPolyCoeffs->GetDeviceInstance();
        m_objects->cu_regressionWeights = m_hostRegressionWeights->GetDeviceInstance();
        m_objects.Upload();

        // Generate some random numbers to seed the polynomial coefficients
        KernelRandomisePolynomialCoefficients << < (m_objects->numPolyCoeffs + 255) / 256, 256, 0, m_hostStream >> > (m_objects->cu_polyCoeffs);
    }

    __global__ void KernelComputeRegressionWeights(Host::LightProbeRegressionFilter::Objects* objectsPtr, const int probeStartIdx)
    {
        __shared__ Host::LightProbeRegressionFilter::Objects objects;
        if (kThreadIdx == 0)
        {
            assert(objectsPtr);
            objects = *objectsPtr;
            assert(objects.gridData.cu_inputGrid);
            assert(objects.gridData.cu_outputGrid);
        }

        __syncthreads();


    }

    __global__ void KernelComputeRegressionIteration(Host::LightProbeRegressionFilter::Objects* objects)
    {
        assert(objects->gridData.cu_inputGrid);
        assert(objects->gridData.cu_outputGrid);

    }

    __device__ __forceinline__ void Modulus(const int composite, int& modulus)
    {
        modulus = composite;
    }

    template<typename... Pack>
    __device__ __forceinline__ void Modulus(const int composite, int& modulus, const int& size, Pack... pack)
    {
        modulus = composite % size;
        Modulus(composite / size, pack...);
    }

    __global__ void KernelReconstructPolynomial(Host::LightProbeRegressionFilter::Objects* objectsPtr)
    {
        __shared__ Host::LightProbeRegressionFilter::Objects objects;
        if (kThreadIdx == 0)
        {
            assert(objectsPtr);
            objects = *objectsPtr;
            assert(objects.gridData.cu_inputGrid);
            assert(objects.gridData.cu_outputGrid);
        }    

        __syncthreads();

        // Probes -> SH coefficients -> Polynomial coefficients + max/min

        const auto& gridData = objects.gridData;
        const auto& params = objects.params;
        const int probeIdx0 = kKernelIdx / gridData.shCoeffsPerProbe;
        const int coeffIdx = kKernelIdx % gridData.shCoeffsPerProbe;

        if (coeffIdx > 0)
        {
            gridData.cu_outputGrid->SetSHCoefficient(probeIdx0, coeffIdx, kZero);
            return;
        }

        const ivec3 pos0 = GridPosFromProbeIdx(probeIdx0, gridData.density);

        vec3 LSum(0.0f);
        int sumWeights = 0;
        for (int z = -params.reconstructionRadius; z <= params.reconstructionRadius; ++z)
        {
            for (int y = -params.reconstructionRadius; y <= params.reconstructionRadius; ++y)
            {
                for (int x = -params.reconstructionRadius; x <= params.reconstructionRadius; ++x)
                {
                    ivec3 posK = pos0 + ivec3(x, y, z);
                    if (gridData.cu_inputGrid->IdxAt(posK) < 0) { continue; }                    

                    const int probeIdxK = ProbeIdxFromGridPos(posK, gridData.density);
                    const int dataIdxK = probeIdxK * objects.polyCoeffsPerProbe + coeffIdx * objects.polyCoeffsPerCoefficient;
                    assert(dataIdxK < objects.cu_polyCoeffs->Size());
                    const vec3* polyCoeffs = &(*objects.cu_polyCoeffs)[dataIdxK];
                    
                    vec3 L(0.0f);
                    float zExp = 1.0f;
                    for (int zt = 0, t = 0; zt <= params.polynomialOrder; zt++)
                    {
                        float yExp = 1.0;
                        for (int yt = 0; yt <= params.polynomialOrder; yt++)
                        {
                            float xExp = 1.0;
                            for (int xt = 0; xt <= params.polynomialOrder; xt++, t++)
                            {
                                L = polyCoeffs[t] * xExp * yExp * zExp;
                                xExp *= x;
                            }
                            yExp *= y;
                        }
                        zExp *= z;
                    }
                    LSum += L;
                    sumWeights += 1;
                }
            }
        }
        
        gridData.cu_outputGrid->SetSHCoefficient(probeIdx0, coeffIdx, LSum / float(sumWeights));
    }

    __host__ void Host::LightProbeRegressionFilter::OnPostRenderPass()
    {
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        // Pass-through filter just copies the data
        if (m_objects->params.isNullFilter)
        {
            m_hostOutputGrid->Replace(*m_hostInputGrid);
            return;
        }

        //KernelComputeRegressionWeights << < (m_objects->gridData.totalSHCoefficients + 255) / 256, 256, 0, m_hostStream >> > (m_objects.GetDeviceObject());

        //KernelComputeRegressionIteration <<< >>> (m_objects.GetDeviceObject());

        KernelReconstructPolynomial << < (m_objects->gridData.totalSHCoefficients + 255) / 256, 256, 0, m_hostStream >> > (m_objects.GetDeviceObject());
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