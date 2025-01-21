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

    static constexpr int kNumHarmonics = 13;
    static constexpr int kW = 4 * kNumHarmonics;
    using InputSample = NNano::Tensor1D<kW>;
    using OutputSample = NNano::Tensor1D<1>;
    
    class NNanoEvaluator : NNano::DataAccessor<InputSample, OutputSample>
    {
    public:
        using ActivationFunction = NNano::Activation::LeakyReLU;
        using LossFunction = NNano::Loss::L2;
        using OptimiserFunction = NNano::Adam<std::ratio<1, 1000>>;

        using Model = NNano::LinearSequential<NNano::Linear<kW, kW>, NNano::Linear<kW, kW>, NNano::Linear<kW, kW>, NNano::Linear<kW, 1>>;
        using ModelInitialiser = NNano::UniformXavierInitialiser;
        using MLP = NNano::MLP<Model, ModelInitialiser, ActivationFunction, LossFunction, OptimiserFunction>;

    private:        
        std::unique_ptr<MLP> m_mlp;
        std::thread         m_workerThread;
        std::atomic<int>    m_threadState;
        std::mutex          m_nnMutex;
        cudaStream_t        m_cudaStream;

        AssetHandle<Host::SDFQuadraticSpline>   m_sdf;
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

        __host__ virtual int LoadTrainingSet(NNano::Cuda::Vector<InputSample>&, NNano::Cuda::Vector<OutputSample>&) override final;
        __host__ virtual int LoadInferenceBatch(NNano::Cuda::Vector<InputSample>&, const int startIdx) override final;
        __host__ virtual void StoreInferenceBatch(const NNano::Cuda::Vector<OutputSample>&, const int startIdx) override final;

    public:
        __host__ NNanoEvaluator(AssetHandle<Host::SDFQuadraticSpline>& sdf, const BBox2f& sdfBounds, const ivec2& gridDims, AssetHandle<Host::DualImage3f>& inferenceImage);
        __host__ virtual ~NNanoEvaluator();

    };
}