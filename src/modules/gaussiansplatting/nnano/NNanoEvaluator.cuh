#pragma once

#include "core/assets/GenericObject.cuh"

#include "../FwdDecl.cuh"
#include <thirdparty/nnano/src/models/mlp/mlp.cuh>
#include <thirdparty/nnano/src/core/cuda/CudaVector.cuh>
#include "SDFQuadraticSpline.cuh"
#include "core/containers/DualImage.cuh"

#include <thread>
#include <mutex>

namespace Enso
{    
    class QuadraticSpline;
    
    namespace Host
    {
        class PDF2;
    }

    static constexpr int kNumHarmonics = 13;
    static constexpr int kW = 4 * kNumHarmonics;
    using InputSample = NNano::Tensor1D<kW>;
    //using InputSample = NNano::Tensor1D<2>;
    using OutputSample = NNano::Tensor1D<1>;
    
    class NNanoEvaluator : public Host::Dirtyable, public NNano::DataAccessor<InputSample, OutputSample>
    {
    public:
        using ActivationFunction = NNano::Activation::LeakyReLU;
        using LossFunction = NNano::Loss::L1;
        using OptimiserFunction = NNano::Adam<std::ratio<1, 5000>>;
        using Model = NNano::LinearSequential<NNano::Linear<kW, kW>, NNano::Linear<kW, kW>, NNano::Linear<kW, kW>, NNano::Linear<kW, 1>>;
        using ModelInitialiser = NNano::UniformXavierInitialiser;

        /*using ActivationFunction = NNano::Activation::Sine;
        using LossFunction = NNano::Loss::L2;
        using OptimiserFunction = NNano::Adam<std::ratio<1, 10000>>;
        using Model = NNano::LinearSequential<NNano::Linear<2, 64>, NNano::Linear<64, 64>, NNano::Linear<64, 64>, NNano::Linear<64, 1>>;
        using ModelInitialiser = NNano::SirenInitialiser;*/

        using MLP = NNano::MLP<Model, ModelInitialiser, ActivationFunction, LossFunction, OptimiserFunction, 256>;

    private:        
        std::unique_ptr<MLP> m_mlp;
        std::thread         m_workerThread;
        std::atomic<int>    m_threadState;
        std::mutex          m_nnMutex;
        cudaStream_t        m_cudaStream;
        int                 m_epochIdx = 0;

        AssetHandle<Host::SDFQuadraticSpline>   m_sdf;
        AssetHandle<Host::PDF2>                 m_hostPDF;
        ivec2                                   m_gridDims;
        BBox2f                                  m_sdfBounds;
        AssetHandle<Host::DualImage3f>          m_inferenceImage;

        enum ThreadSignal : int
        {
            kThreadUndefined = 0,
            kThreadRunning,
            kThreadShutdown,
            kThreadExpired
        };

    private:
        __host__ void Run();

        __host__ virtual int LoadInferenceBatch(NNano::Cuda::Vector<InputSample>&, const int startIdx) override final;
        __host__ virtual void StoreInferenceBatch(const NNano::Cuda::Vector<OutputSample>&, const int startIdx) override final;
        __host__ void PrepareEpoch(const int epochIdx);
        __host__ void PreparePDF();

    public:
        __host__ NNanoEvaluator(const InitCtx& initCtx, AssetHandle<Host::SDFQuadraticSpline>& sdf, const BBox2f& sdfBounds, const ivec2& gridDims, AssetHandle<Host::DualImage3f>& inferenceImage);
        __host__ virtual ~NNanoEvaluator();

    };
}