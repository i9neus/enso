#include "NNanoEvaluator.cuh"
#include "core/utils/HighResolutionTimer.h"
#include "core/math/samplers/PCG.cuh"

namespace Enso
{
    __device__ __forceinline__ InputSample PositionalEncode(const vec2& p)
    {
        // Compute the Fourier positional encoding
        InputSample sample;
        for (int harmonic = 0; harmonic < kNumHarmonics; ++harmonic)
        {
            for (int d = 0; d < 4; ++d)
            {
                sample[4 * harmonic + d] = sinf(kPi * ((1 + harmonic) * p[d >> 1] + float(d & 1)) / 2);
            }
        }
        return sample;
    }   

    __global__ void KernelGenerateInferenceSet(const Device::SDFQuadraticSpline* sdf, InputSample* inputData, const BBox2f sdfBounds, const ivec2 setDims)
    {
        CudaAssertDebug(sdf);
        CudaAssertDebug(inputData || targetData);

        const auto kernelIdx = kKernelIdx;
        if (kernelIdx < Area(setDims))
        {
            const vec2 p = TransformNormalisedScreenToView(vec2(float(kernelIdx % setDims.x) / float(setDims.x), float(kernelIdx / setDims.x) / float(setDims.y)), setDims);
            
            //inputData[kernelIdx] = PositionalEncode(p);
            inputData[kernelIdx] = InputSample({ p.x, p.y });
        }
    }
    
    __global__ void KernelGenerateTrainingSet(const Device::SDFQuadraticSpline* sdf, InputSample* inputData, OutputSample* targetData, const BBox2f sdfBounds, const int numSamples, const int seed)
    {
        CudaAssertDebug(sdf);
        CudaAssertDebug(inputData || targetData);

        const auto kernelIdx = kKernelIdx;
        if (kernelIdx < numSamples)
        {
            // Generate random numbers in the range [0, 1]
            const vec4 xi = PCG(HashOf(kernelIdx, seed)).Rand();
            const vec2 p = TransformNormalisedScreenToView(xi.xy, sdfBounds.Dimensions());

            //inputData[kernelIdx] = PositionalEncode(p);
            inputData[kernelIdx] = InputSample({ p.x, p.y });
            targetData[kernelIdx] = NNano::Activation::TanH::F(sdf->Evaluate(p).x);
        }
    }

    template<typename SampleType>
    __global__ void KernelRenderTensorField(const SampleType* targetData, Device::DualImage3f* image, const int offset)
    {
        CudaAssertDebug(targetData);
        CudaAssertDebug(image);

        const auto kernelIdx = kKernelIdx;
        if (kernelIdx < image->Area())
        {
            // Evaluate the value of the SDF
            //const float f = cosf(10.f * kTwoPi * targetData[kernelIdx][Offset]) * 0.5f + 0.5f;
            float f = NNano::Activation::TanH::InvF(targetData[kernelIdx][offset]);

            f = cosf(10.f * kTwoPi * f) * 0.5f + 0.5f;

            // Write to the buffer
            *reinterpret_cast<vec3*>(image->At(kernelIdx % image->Width(), kernelIdx / image->Width())) = f;
        }
    }
    
    __host__ NNanoEvaluator::NNanoEvaluator(const InitCtx& initCtx, AssetHandle<Host::SDFQuadraticSpline>& sdf, const BBox2f& sdfBounds, const ivec2& gridDims, AssetHandle<Host::DualImage3f>& inferenceImage) :
        Host::Dirtyable(initCtx),
        m_threadState(kThreadRunning),
        m_workerThread(&NNanoEvaluator::Run, this),
        m_sdf(sdf),
        m_sdfBounds(sdfBounds),
        m_gridDims(gridDims),
        m_inferenceImage(inferenceImage)
    {
        // Create a new stream that will run all NNano operations for this model
        IsOk(cudaStreamCreate(&m_cudaStream));

        // Create the model
        m_mlp.reset(new MLP(*this, m_cudaStream));
    }

    __host__ int NNanoEvaluator::LoadInferenceBatch(NNano::Cuda::Vector<InputSample>& inputSamples, const int startIdx)
    {
        if (startIdx != 0) { return 0; }

        const auto numSamples = Area(m_gridDims);
        inputSamples.Resize(numSamples);

        auto [gridDims, blockDims] = Get1DLaunchParams(numSamples);
        KernelGenerateInferenceSet << <gridDims, blockDims, 0, m_cudaStream >> > (m_sdf->GetDeviceInstance(), inputSamples.GetComputeData(), m_sdfBounds, m_gridDims);
        IsOk(cudaStreamSynchronize(m_cudaStream));
        
        return numSamples;
    }

    __host__ void NNanoEvaluator::StoreInferenceBatch(const NNano::Cuda::Vector<OutputSample>& outputSamples, const int startIdx)
    {
        //auto [inputSamples, targetSamples] = m_mlp->GetTrainingSet();
        //auto& inputSamples = *m_mlp->GetInferenceSet();
        
        Assert(outputSamples.Size() == Area(m_gridDims));

        auto [gridDims, blockDims] = Get1DLaunchParams(Area(m_gridDims));
        KernelRenderTensorField<< <gridDims, blockDims, 0, m_cudaStream >> > (outputSamples.GetComputeData(), m_inferenceImage->GetDeviceInstance(), 0);
        IsOk(cudaStreamSynchronize(m_cudaStream));
    }

    __host__ NNanoEvaluator::~NNanoEvaluator()
    {
        if (m_threadState == kThreadRunning)
        {
            Log::Debug("Shutting down NNano...");
            
            constexpr int kWorkerThreadTimeoutMs = 5000;
            m_threadState = kThreadShutdown;
            int waitInterval = 32;
            int accumTime = 0;
            do
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(waitInterval));
                accumTime += waitInterval;
                waitInterval *= 2;
                if (accumTime >= kWorkerThreadTimeoutMs)
                {
                    Log::Error("Error: NNano evaluator stopped responding.");
                    m_workerThread.detach();
                    break;
                }
            } 
            while (m_threadState != kThreadExpired);

            // If the thread has been successfully cleaned up, destroy the stream
            if (m_threadState == kThreadExpired) { cudaStreamDestroy(m_cudaStream); }
        }
    }

    __host__ void NNanoEvaluator::PrepareEpoch(const int epochIdx)
    {
        auto [inputSamples, targetSamples] = m_mlp->GetTrainingDataObjects();
        
        const int kNumSamples = MLP::kMiniBatchSize * 50;
        inputSamples->Resize(kNumSamples);
        targetSamples->Resize(kNumSamples);

        auto [gridDims, blockDims] = Get1DLaunchParams(kNumSamples, 256);
        KernelGenerateTrainingSet << <gridDims, blockDims, 0, m_cudaStream >> > (m_sdf->GetDeviceInstance(), inputSamples->GetComputeData(), targetSamples->GetComputeData(), m_sdfBounds, kNumSamples, epochIdx);
        IsOk(cudaStreamSynchronize(m_cudaStream));

        m_mlp->PrepareTraining();
    }

    __host__ void NNanoEvaluator::Run()
    {
        Log::Write("Running continuous NNano training...");

        m_mlp->Initialise();
        
        HighResolutionTimer inferenceTimer;
        for(int epochIdx = 0; m_threadState == kThreadRunning; ++epochIdx)
        {
            PrepareEpoch(epochIdx);
            
            m_mlp->TrainEpoch();

            if (inferenceTimer.Get() > 0.2f)
            {
                m_mlp->Infer();
                
                auto stats = m_mlp->GetStats();
                Log::Debug("Epoch %i: loss %.4f", stats.numEpochs, stats.loss);
                
                inferenceTimer.Reset();
                SignalDirty(kDirtyViewportRedraw);
                Clean();
            }
        }

        Log::Debug("Done!");
    }
}