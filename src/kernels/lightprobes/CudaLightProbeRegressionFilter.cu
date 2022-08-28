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
        regressionIterations(1),
        learningRate(0.005f), 
        minSamples(64),
        regressionMaxIterations(-1),
        tikhonovCoeff(0.0f)
    {

    }

    __host__ void LightProbeRegressionFilterParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("polynomialOrder", polynomialOrder);
        node.AddValue("regressionRadius", regressionRadius);
        node.AddValue("regressionIterations", regressionIterations);
        node.AddValue("reconstructionRadius", reconstructionRadius);
        node.AddValue("learningRate", learningRate);
        node.AddValue("minSamples", minSamples);
        node.AddValue("tikhonovCoeff", tikhonovCoeff);
        node.AddValue("regressionMaxIterations", regressionMaxIterations);
        node.AddEnumeratedParameter("filterType", std::vector<std::string>({ "null", "box", "gaussian", "nlm" }), filterType);

        Json::Node nlmNode = node.AddChildObject("nlm");
        nlm.ToJson(nlmNode);
    }

    __host__ uint LightProbeRegressionFilterParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("polynomialOrder", polynomialOrder, flags);
        node.GetValue("regressionRadius", regressionRadius, flags);
        node.GetValue("regressionIterations", regressionIterations, flags);
        node.GetValue("reconstructionRadius", reconstructionRadius, flags);
        node.GetValue("learningRate", learningRate, flags);
        node.GetValue("minSamples", minSamples, flags);
        node.GetValue("tikhonovCoeff", tikhonovCoeff, flags);
        node.GetValue("regressionMaxIterations", regressionMaxIterations, flags);
        node.GetEnumeratedParameter("filterType", std::vector<std::string>({ "null", "box", "gaussian", "nlm" }), filterType, flags);

        Json::Node nlmNode = node.GetChildObject("nlm", flags);
        if (nlmNode) { nlm.FromJson(nlmNode, flags); }

        regressionRadius = clamp(regressionRadius, 0, 10);
        regressionIterations = clamp(regressionIterations, 1, 100);
        reconstructionRadius = clamp(reconstructionRadius, 0, 10);
        polynomialOrder = clamp(polynomialOrder, 0, 3);

        return kRenderObjectDirtyRender;
    }

    __host__ Host::LightProbeRegressionFilter::LightProbeRegressionFilter(const std::string& id, const ::Json::Node& node) :
        RenderObject(id)
    {
        FromJson(node, Json::kSilent);

        node.GetValue("inputGridID", m_inputGridID, Json::kRequiredAssert);
        node.GetValue("crossGridID", m_crossGridID, Json::kNotBlank);
        node.GetValue("crossGridHalfID", m_crossGridHalfID, Json::kNotBlank);
        node.GetValue("outputGridID", m_outputGridID, Json::kRequiredAssert);

        AssertMsgFmt(!GlobalResourceRegistry::Get().Exists(m_outputGridID), "Error: an asset with ID '%s' already exists'.", m_outputGridID.c_str());

        // Create the output grid
        m_hostOutputGrid = CreateChildAsset<Host::LightProbeGrid>(m_outputGridID);

        // Create the buffers used by the regressor
        m_hostC = CreateChildAsset<Host::Array<vec3>>(tfm::format("%s_C", id), m_hostStream);
        m_hostD = CreateChildAsset<Host::Array<float>>(tfm::format("%s_D", id), m_hostStream);
        m_hostdLdC = CreateChildAsset<Host::Array<vec3>>(tfm::format("%s_dLdC", id), m_hostStream);
        m_hostW = CreateChildAsset<Host::Array<float>>(tfm::format("%s_regressionWeights", id), m_hostStream);

        // TODO: Make weight map dynamic
        m_hostW->Resize(1024 * 1024);
    }

    __host__ AssetHandle<Host::RenderObject> Host::LightProbeRegressionFilter::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kLightProbeFilter) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::LightProbeRegressionFilter>(id, json);
    }

    __host__ uint Host::LightProbeRegressionFilter::FromJson(const ::Json::Node& node, const uint flags)
    {
        m_objects->params.FromJson(node, flags);

        Prepare();

        return kRenderObjectDirtyRender;
    }

    __host__ void Host::LightProbeRegressionFilter::OnDestroyAsset()
    {
        m_hostOutputGrid.DestroyAsset();
        m_hostC.DestroyAsset();
        m_hostD.DestroyAsset();
        m_hostdLdC.DestroyAsset();
        m_hostW.DestroyAsset();
    }

    __host__ void Host::LightProbeRegressionFilter::Bind(RenderObjectContainer& sceneObjects)
    {
        m_hostInputGrid = sceneObjects.FindByID(m_inputGridID).DynamicCast<Host::LightProbeGrid>();
        if (!m_hostInputGrid)
        {
            Log::Error("Error: LightProbeRegressionFilter::Bind(): the specified input light probe grid '%s' is invalid.\n", m_inputGridID);
            return;
        }
        else
        {
            Log::Write("Bound input light probe grid '%s' to regression filter.", m_inputGridID);
        }

        m_hostCrossGrid = nullptr;
        if (!m_crossGridID.empty())
        {
            m_hostCrossGrid = sceneObjects.FindByID(m_crossGridID).DynamicCast<Host::LightProbeGrid>();
            if (!m_hostCrossGrid)
            {
                Log::Error("Error: LightProbeRegressionFilter::Bind(): the specified cross light probe grid '%s' is invalid.\n", m_crossGridID);
                return;
            }
            else
            {
                Log::Write("Bound cross light probe grid '%s' to regression filter.", m_crossGridID);
            }
        }

        m_hostCrossHalfGrid = nullptr;
        if (!m_crossGridHalfID.empty())
        {
            m_hostCrossHalfGrid = sceneObjects.FindByID(m_crossGridHalfID).DynamicCast<Host::LightProbeGrid>();
            if (!m_hostCrossHalfGrid)
            {
                Log::Error("Error: LightProbeRegressionFilter::Bind(): the specified half cross light probe grid '%s' is invalid.\n", m_crossGridHalfID);
                return;
            }
            else
            {
                Log::Write("Bound half cross light probe grid '%s' to regression filter.", m_crossGridHalfID);
            }
        }

        Prepare();
    }

    __global__ void KernelRandomisePolynomialCoefficients(Device::Array<vec3>* coeffBuffer)
    {
        assert(coeffBuffer);
        assert(kKernelIdx < coeffBuffer->Size());

        PseudoRNG rng(HashOf(uint(kKernelIdx)));
        (*coeffBuffer)[kKernelIdx] = rng.Rand<0>() * 0.5f - 0.25f;
    }

    __host__ void Host::LightProbeRegressionFilter::Prepare()
    {
        m_iterationCount = 0;
        
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        Assert(m_hostC); // Sanity check

        // Establish the dimensions of the kernel
        auto& gridData = m_objects->gridData.Initialise(m_hostInputGrid, m_hostCrossHalfGrid,
                                                        m_hostCrossGrid, m_hostCrossHalfGrid, 
                                                        m_hostOutputGrid, AssetHandle<Host::LightProbeGrid>(nullptr));
        Assert(m_objects->gridData.coefficientsPerProbe <= kMaxCoefficients);
     
        // Precompute some values for the regression step
        m_objects->regression.radius = m_objects->params.regressionRadius;
        m_objects->regression.span = 2 * m_objects->regression.radius + 1;
        m_objects->regression.volume = cub(m_objects->regression.span);
        m_objects->regression.numMonomials = cub(m_objects->params.polynomialOrder + 1);

        if (m_objects->regression.volume < m_objects->regression.numMonomials)
        {
            Log::Error("Warning: the regression system with %i coeffients is under-determined given the window size of %i probes", m_objects->regression.numMonomials, m_objects->regression.volume);
        }

        // Precompute some values for the reconstruction step
        m_objects->reconstruction.radius = m_objects->params.reconstructionRadius;
        m_objects->reconstruction.span = 2 * m_objects->reconstruction.radius + 1;
        m_objects->reconstruction.volume = cub(m_objects->reconstruction.span);
        
        // Each coefficient stores the monomial coefficients of the fitted polynomial plus 2 additional coefficients which represent the max and min values of the kernel   
        m_objects->regrCoeffsPerSHCoeff = m_objects->regression.numMonomials + 2;
        m_objects->regrCoeffsPerProbe = m_objects->regrCoeffsPerSHCoeff * gridData.coefficientsPerProbe;
        m_objects->totalRegrCoeffs = m_objects->regrCoeffsPerProbe * gridData.numProbes;

        // Rather than store every weight for every probe, we do the regression step in batches to cap the memory needed
        m_objects->probesPerBatch = std::min(gridData.numProbes, int(m_hostW->Size() / m_objects->regression.volume));
        Log::Debug("Probes per batch: %i", m_objects->probesPerBatch);

        // Resize the polynomial coefficient array as a power of two 
        m_hostdLdC->ExpandToNearestPow2(m_objects->totalRegrCoeffs);
        if (m_hostC->ExpandToNearestPow2(m_objects->totalRegrCoeffs))
        {
            Log::Debug("Resized C/dLdC to %i", m_hostC->Size());
        }

        // Generate a precomputed set of matrices containing the monomial constants for the kernel
        PrecomputeMonomialMatrices();

        // Initialise the output grid so it has the same dimensions as the input
        m_hostOutputGrid->Prepare(m_hostInputGrid->GetParams());

        // Set the device objects and sync
        m_objects->cu_C = m_hostC->GetDeviceInstance();
        m_objects->cu_dLdC = m_hostdLdC->GetDeviceInstance();
        m_objects->cu_D = m_hostD->GetDeviceInstance();
        m_objects->cu_W = m_hostW->GetDeviceInstance();
        m_objects->cu_T = m_hostT.GetDeviceInstance();
        m_objects.Upload();

        // Generate some random numbers to seed the polynomial coefficients
        KernelRandomisePolynomialCoefficients << < (m_objects->totalRegrCoeffs + 255) / 256, 256, 0, m_hostStream >> > (m_objects->cu_C);

        // Compute initialisation data for the regression kernel
        auto& rk = m_regressionKernel; 
        constexpr int kMaxSharedMemory = 40 * 1024;
        constexpr int kMinBlockSize = 8;
        rk.blockSize = 256;
        do
        {
            rk.sharedMemoryBytes = m_objects->regression.volume * sizeof(float) * rk.blockSize;
            rk.blockSize >>= 1;
        }
        while(rk.sharedMemoryBytes >= kMaxSharedMemory && rk.blockSize >= kMinBlockSize);
        Assert(rk.blockSize >= kMinBlockSize);

        rk.gridSize = ((m_objects->probesPerBatch * m_objects->gridData.shCoeffsPerProbe * 3) + (rk.blockSize - 1)) / rk.blockSize;

        Log::Debug("Regression kernel:");
        Log::Debug("  - Grid size: %i", rk.gridSize);
        Log::Debug("  - Block size: %i", rk.blockSize);
        Log::Debug("  - Shared memory: %i", rk.sharedMemoryBytes);
    }

    __host__ void Host::LightProbeRegressionFilter::PrecomputeMonomialMatrices()
    {
        // Allocate some temporary memory
        const int numElements = m_objects->regression.volume * m_objects->regression.numMonomials;
        std::vector<float> D(numElements);
        const int radius = m_objects->regression.radius;
        const int polynomialOrder = m_objects->params.polynomialOrder;

        // Generate a monomial matrix for every point in the regression kernel. 
        // TODO: There's some redundancy here, but we can see to that later.
        for (int z = -radius, dIdx = 0; z <= radius; ++z)
        {
            const float nz = z / float(::max(1, radius));
            for (int y = -radius; y <= radius; ++y)
            {
                const float ny = y / float(::max(1, radius));
                for (int x = -radius; x <= radius; ++x)
                {
                    // Construct the monomial matrix
                    const float nx = x / float(::max(1, radius));
                    float zExp = 1.0f;
                    for (int zt = 0, cIdx = 0; zt <= polynomialOrder; zt++)
                    {
                        float yExp = 1.0f;
                        for (int yt = 0; yt <= polynomialOrder; yt++)
                        {
                            float xExp = 1.0f;
                            for (int xt = 0; xt <= polynomialOrder; ++xt)
                            {
                                D[dIdx++] = xExp * yExp * zExp;
                                xExp *= nx;
                            }
                            yExp *= ny;
                        }
                        zExp *= nz;
                    }
                }
            }
        }
        // Upload the matrices to the Cuda array
        m_hostD->Upload(D);

        // Compute the constants for the Tikhonov regularisation matrix
        for (int zt = 0, tIdx = 0; zt <= polynomialOrder; zt++)
        {
            float yExp = 1.0f;
            for (int yt = 0; yt <= polynomialOrder; yt++)
            {
                float xExp = 1.0f;
                for (int xt = 0; xt <= polynomialOrder; ++xt, ++tIdx)
                {
                    //m_hostT[tIdx] = (xt + yt + zt) * m_objects->params.tikhonovCoeff;
                    m_hostT[tIdx] = (tIdx == 0) ? 0.0f : m_objects->params.tikhonovCoeff;
                }
            }
        }
        m_hostT.Upload();
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

        // Get the index of the probe in the grid and the sample in the kernel
        auto& gridData = objects.gridData;
        const auto& params = objects.params;

        // Compute the index and position of the probe and bail out if we're out of bounds
        const int probeIdx0 = probeStartIdx + kKernelIdx / objects.regression.volume;
        if (probeIdx0 >= gridData.numProbes) { return; }
        const ivec3 pos0 = GridPosFromProbeIdx(probeIdx0, objects.gridData.density);

        // Compute the sample position relative to the centre of the kernel
        const int probeIdxK = kKernelIdx % objects.regression.volume;
        const ivec3 posK = GridPosFromProbeIdx(probeIdxK, objects.regression.span) - ivec3(objects.regression.radius);

        // If the neighbourhood probe lies outside the bounds of the grid, set the weight to zero
        const int probeIdxN = gridData.cu_inputGrid->IdxAt(pos0 + posK);
        if (probeIdxN < 0 || 
            gridData.cu_inputGrid->At(probeIdxN)[objects.gridData.coefficientsPerProbe - 1].x < 0.5f)
        {
            (*objects.cu_W)[kKernelIdx] = 0.0f;
        }
        else
        {   
            // Calculate the weight for the sample
            float weight = 0.0f;
            switch (params.filterType)
            {
            case kKernelFilterGaussian:
            {
                const float len = length(vec3(posK));
                if (len <= objects.regression.radius)
                {
                    weight = Integrate1DGaussian(len - 0.5f, len + 0.5f, objects.regression.radius);
                }
                break;
            }
            case kKernelFilterNLM:
            {
                weight = ComputeNLMWeight(gridData, params.nlm, pos0, posK);
                break;
            }
            default:
                weight = 1.0f;
            };

            (*objects.cu_W)[kKernelIdx] = weight;
        }
    }

    __global__ void KernelComputeRegressionIteration(Host::LightProbeRegressionFilter::Objects* objectsPtr, const int probeStartIdx)
    {
        /*
            p -> pixel values i.e. what we're regressing onto
            C -> polynomial coefficients
            D -> monomial constants over the spread of the kernel
            W -> kernel weights
        */

        extern __shared__ int __block[];
        float* pBlock = reinterpret_cast<float*>(__block);
        __shared__ Host::LightProbeRegressionFilter::Objects objects;
        __shared__  const float* D;
        __shared__ float gamma;
        __shared__ float T[4*4*4];

        if (kThreadIdx == 0)
        {
            assert(objectsPtr);
            objects = *objectsPtr;
            assert(objects.gridData.cu_inputGrid);
            assert(objects.gridData.cu_outputGrid);
            assert(objects.cu_D);
            assert(objects.cu_C);
            assert(objects.cu_W);

            D = objects.cu_D->GetData();
            gamma = objects.params.tikhonovCoeff;
            memcpy(T, objects.cu_T, sizeof(float) * 4*4*4);
        }

        __syncthreads();

        // Get a pointer to the shared memory used to cache the kernel values for this channel
        float* p = &pBlock[kThreadIdx * objects.regression.volume];

        const auto& gridData = objects.gridData;
        const auto& params = objects.params;

        const int probeIdx0 = probeStartIdx + kKernelIdx / (gridData.shCoeffsPerProbe * 3);
        if (probeIdx0 >= gridData.numProbes) { return; }

        const int coeffIdx = (kKernelIdx / 3) % gridData.shCoeffsPerProbe;
        const int channelIdx = kKernelIdx % 3;        

        // Get pointers to the polynomial coefficients and associated partial derivatives
        int dataIdx0 = probeIdx0 * objects.regrCoeffsPerProbe + coeffIdx * objects.regrCoeffsPerSHCoeff;
        assert(dataIdx0 < objects.cu_C->Size());
        vec3* C = &(*objects.cu_C)[dataIdx0];
        vec3* dLdC = &(*objects.cu_dLdC)[dataIdx0];
        // Get a pointer to the weights
        float* W = &(*objects.cu_W)[(kKernelIdx / (gridData.shCoeffsPerProbe * 3)) * objects.regression.volume];

        // Fill the cache with the local pixel values
        float sumW = 0.0f;
        float maxP = -kFltMax, minP = kFltMax;
        const ivec3 pos0 = GridPosFromProbeIdx(probeIdx0, gridData.density);
        for (int z = -objects.regression.radius, pIdx = 0; z <= objects.regression.radius; ++z)
        {
            for (int y = -objects.regression.radius; y <= objects.regression.radius; ++y)
            {
                for (int x = -objects.regression.radius; x <= objects.regression.radius; ++x, pIdx++)
                {
                    ivec3 posK = pos0 + ivec3(x, y, z);
                    if (gridData.cu_inputGrid->IdxAt(posK) < 0)
                    {
                        p[pIdx] = -kFltMax;
                        continue;
                    }
                    p[pIdx] = gridData.cu_inputGrid->At(posK)[coeffIdx][channelIdx];
                    maxP = fmaxf(maxP, p[pIdx]);
                    minP = fminf(minP, p[pIdx]);
                    sumW += W[pIdx];
                }
            }
        }

        // Normalise the p-values
        for (int pIdx = 0; pIdx < objects.regression.volume; ++pIdx)
        {
            if (p[pIdx] != -kFltMax)
            {
                p[pIdx] = (p[pIdx] - minP) / fmaxf(1e-5f, maxP - minP);
            }
        }

        // Do the polynomial regression
        for (int itIdx = 0; itIdx < params.regressionIterations; ++itIdx)
        {
            // Clear the loss and derivatives
            float L2Loss = 0.0f;
            for (int t = 0; t < objects.regression.numMonomials; t++) { dLdC[t][channelIdx] = 0.0f; }

            // Loop over every element in the kernel and compute partial derivatives ready for the gradient descent
            for (int pIdx = 0; pIdx < objects.regression.volume; ++pIdx)
            {
                if (p[pIdx] == -kFltMax) { continue; }

                // Accumulate the sum of polynomial coefficients multiplied by the associated monomial constants
                float sigma = -p[pIdx];
                float tikhonov = 0.0f;
                const float& w = W[pIdx];
                for (int cIdx = 0, dIdx = pIdx * objects.regression.numMonomials; cIdx < objects.regression.numMonomials; ++cIdx, ++dIdx)
                {
                    assert(dIdx < objects.cu_D->Size());
                    const float cd = C[cIdx][channelIdx] * D[dIdx];
                    tikhonov += sqr(cd * T[cIdx]);
                    sigma += cd;
                }

                // Accumulate the partial derivatives for each constant
                for (int cIdx = 0, dIdx = pIdx * objects.regression.numMonomials; cIdx < objects.regression.numMonomials; ++cIdx, ++dIdx)
                {
                    dLdC[cIdx][channelIdx] += (2.0 * D[dIdx] * sigma + 
                                               2.0f * C[cIdx][channelIdx] * sqr(D[dIdx] * T[cIdx])) * w;
                }

                // Accumulate the weighted sum of the derivatives as the L2 loss
                L2Loss += (sqr(sigma) + tikhonov) * w;
            }
            L2Loss /= sumW;

            // Perform the gradient descent step
            for (int cIdx = 0; cIdx < objects.regression.numMonomials; ++cIdx)
            {
                dLdC[cIdx][channelIdx] /= sumW;
                C[cIdx][channelIdx] -= params.learningRate * dLdC[cIdx][channelIdx] / fmaxf(L2Loss, 1e-2f);
            }
        }

        // Update the min/max components 
        C[objects.regression.numMonomials][channelIdx] = minP;
        C[objects.regression.numMonomials + 1][channelIdx] = maxP;
    }

    __global__ void KernelReconstructPolynomial(Host::LightProbeRegressionFilter::Objects* objectsPtr)
    {
        __shared__ Host::LightProbeRegressionFilter::Objects objects;
        __shared__  const float* D;
        if (kThreadIdx == 0)
        {
            assert(objectsPtr);
            objects = *objectsPtr;
            assert(objects.gridData.cu_inputGrid);
            assert(objects.gridData.cu_outputGrid);
            
            // Get the monomial matrix at the centre of the 
            D = &(objects.cu_D->GetData()[(objects.regression.volume / 2) * objects.regression.numMonomials]);
        }

        __syncthreads();

        // Probes -> SH coefficients -> Polynomial coefficients + max/min

        const auto& gridData = objects.gridData;
        const auto& params = objects.params;
        const int probeIdx0 = kKernelIdx / gridData.shCoeffsPerProbe;
        if (probeIdx0 >= objects.gridData.numProbes) { return; }
        const int coeffIdx = kKernelIdx % gridData.shCoeffsPerProbe;

        if (coeffIdx > 0)
        {
            gridData.cu_outputGrid->SetSHCoefficient(probeIdx0, coeffIdx, kZero);
            return;
        }

        const ivec3 pos0 = GridPosFromProbeIdx(probeIdx0, gridData.density);

        vec3 LSum(0.0f);
        vec3 lapSum(0.0f);
        int sumWeights = 0;
        float radiusNorm = fmaxf(1, objects.regression.radius);
        for (int z = -objects.reconstruction.radius; z <= objects.reconstruction.radius; ++z)
        {
            const float nz = -z / radiusNorm;
            for (int y = -objects.reconstruction.radius; y <= objects.reconstruction.radius; ++y)
            {
                const float ny = -y / radiusNorm;
                for (int x = -objects.reconstruction.radius; x <= objects.reconstruction.radius; ++x)
                {
                    const float nx = -x / radiusNorm;
                    ivec3 posK = pos0 + ivec3(x, y, z);
                    if (gridData.cu_inputGrid->IdxAt(posK) < 0) { continue; }

                    const int probeIdxK = ProbeIdxFromGridPos(posK, gridData.density);
                    const int dataIdxK = probeIdxK * objects.regrCoeffsPerProbe + coeffIdx * objects.regrCoeffsPerSHCoeff;
                    assert(dataIdxK < objects.cu_C->Size());
                    const vec3* C = &(*objects.cu_C)[dataIdxK];
                    
                    vec3 L(0.0f);
                    vec3 lap(0.0f);
                    int t = 0;
                    float zExp = 1.0f;
                    for (int zt = 0; zt <= params.polynomialOrder; zt++)
                    {
                        float yExp = 1.0;
                        for (int yt = 0; yt <= params.polynomialOrder; yt++)
                        {
                            float xExp = 1.0;
                            for (int xt = 0; xt <= params.polynomialOrder; xt++, t++)
                            {
                                L += C[t] * xExp * yExp * zExp;
                                if (xt == 2) { lap += 2.0f * C[t] * yExp * zExp; }
                                if (yt == 2) { lap += 2.0f * C[t] * xExp * zExp; }
                                if (zt == 2) { lap += 2.0f * C[t] * xExp * yExp; }
                                xExp *= nx;
                            }
                            yExp *= ny;
                        }
                        zExp *= nz;
                    }

                    // Denormalise
                    const vec3& kernelMin = C[t];
                    const vec3& kernelMax = C[t + 1];
                    L = kernelMin + L * (kernelMax - kernelMin);
                    lap = (lap * (kernelMax - kernelMin)) / max(vec3(1e-2f), L);
                    
                    // Accumulate
                    LSum += L;
                    lapSum += lap;
                    sumWeights += 1;
                }
            }
        }
        
        gridData.cu_outputGrid->SetSHCoefficient(probeIdx0, coeffIdx, LSum / float(sumWeights));
        gridData.cu_outputGrid->SetSHLaplacianCoefficient(probeIdx0, coeffIdx, lapSum / float(sumWeights));
    }

    __host__ void Host::LightProbeRegressionFilter::OnPostRenderPass()
    {
        // Filter isn't yet bound, so do nothing
        if (!m_hostInputGrid || !m_hostOutputGrid) { return; }

        // If the input has changed, update the output to match
        if (m_hostInputGrid->GetParams() != m_hostOutputGrid->GetParams())
        {
            Prepare();
        }

        // Pass-through filter just copies the data
        if (m_objects->params.filterType == kKernelFilterNull || !m_hostInputGrid->IsConverged()) 
        {
            m_iterationCount = 0;
            m_hostOutputGrid->Replace(*m_hostInputGrid);
            return;
        }

        // If the filter has exceeded the maximum number of iterations, we're done
        if (m_objects->params.regressionMaxIterations > 0 && m_iterationCount >= m_objects->params.regressionMaxIterations) { return; }        

        for (int probeStartIdx = 0; probeStartIdx < m_objects->gridData.numProbes; probeStartIdx += m_objects->probesPerBatch)
        {
            // Populate the kernel weights buffer ready for the regression step
            int numElements = m_objects->probesPerBatch * m_objects->regression.volume;
            KernelComputeRegressionWeights << < (numElements + 255) / 256, 256, 0, m_hostStream >> > (m_objects.GetDeviceInstance(), probeStartIdx);

            // Run the regression step
            KernelComputeRegressionIteration <<< m_regressionKernel.gridSize, m_regressionKernel.blockSize, m_regressionKernel.sharedMemoryBytes, m_hostStream >>> (m_objects.GetDeviceInstance(), probeStartIdx);
        }       

        KernelReconstructPolynomial << < (m_objects->gridData.totalSHCoefficients + 255) / 256, 256, 0, m_hostStream >> > (m_objects.GetDeviceInstance());

        m_hostOutputGrid->UpdateAggregateStatistics(0);

        m_iterationCount += m_objects->params.regressionIterations;
    }

    __host__ std::vector<AssetHandle<Host::RenderObject>> Host::LightProbeRegressionFilter::GetChildObjectHandles()
    {
        std::vector<AssetHandle<Host::RenderObject>> objects;
        objects.emplace_back(m_hostOutputGrid);
        return objects;
    }

    __host__ void Host::LightProbeRegressionFilter::OnUpdateSceneGraph(RenderObjectContainer& sceneObjects, const uint dirtyFlags)
    {
        if (m_hostInputGrid && m_hostOutputGrid &&
            m_hostInputGrid->GetParams() != m_hostOutputGrid->GetParams())
        {
            Prepare();
        }
    }

    __host__ bool Host::LightProbeRegressionFilter::EmitStatistics(Json::Node& rootNode) const
    {
        Json::Node gridSetNode = rootNode.AddChildObject("grids");

       const auto& stats = m_hostOutputGrid->GetAggregateStatistics();
       
       Json::Node gridNode = gridSetNode.AddChildObject(m_outputGridID);
       gridNode.AddValue("minSamples", int(stats.minMaxSamples.x));
       gridNode.AddValue("maxSamples", int(stats.minMaxSamples.y));
       gridNode.AddValue("meanProbeValidity", stats.meanValidity);
       gridNode.AddValue("meanProbeDistance", stats.meanDistance);

       std::vector<std::vector<uint>> histogramData(4);
       for (int idx = 0; idx < 4; ++idx)
       {
           histogramData[idx].resize(50);
           std::memcpy(histogramData[idx].data(), &stats.coeffHistogram[50 * idx], sizeof(uint) * 50);
       }
       gridNode.AddArray2D("coeffHistograms", histogramData);

       return true;
    }
}