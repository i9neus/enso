#include "NNanoEvaluator.cuh"

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
                sample[4 * harmonic + d] = sinf(kPi * (1 + harmonic) * (p[d >> 1] + 0.5f * float(d & 1)));
            }
        }
        return sample;
    }   
    
    __global__ void KernelGenerateSDFDataset(const Device::SDFQuadraticSpline* sdf, InputSample* inputData, OutputSample* targetData, const BBox2f sdfBounds, const ivec2 setDims)
    {
        CudaAssertDebug(sdf);
        CudaAssertDebug(inputData || targetData);

        const auto kernelIdx = kKernelIdx;
        if (kernelIdx < Area(setDims))
        {
            const vec2 p(mix(sdfBounds[0].x, sdfBounds[1].x, (0.5f + kernelIdx % setDims.x) / float(setDims.x)),
                         mix(sdfBounds[0].y, sdfBounds[1].y, (0.5f + kernelIdx / setDims.x) / float(setDims.y)));

            if (inputData)
            {
                inputData[kernelIdx] = PositionalEncode(p);
            }
            if (targetData)
            {
                targetData[kernelIdx] = sdf->Evaluate(p).x;
            }
        }
    }

    __global__ void KernelRenderInference(const OutputSample* targetData, Device::DualImage3f* image)
    {
        CudaAssertDebug(targetData);
        CudaAssertDebug(image);

        const auto kernelIdx = kKernelIdx;
        if (kernelIdx < image->Area())
        {
            // Evaluate the value of the SDF
            const float f = cosf(10.f * kTwoPi * targetData[kernelIdx][0]) * 0.5f + 0.5f;
            // Write to the buffer
            *reinterpret_cast<vec3*>(image->At(kernelIdx % image->Width(), kernelIdx / image->Width())) = f;
        }
    }
    
    __host__ NNanoEvaluator::NNanoEvaluator(AssetHandle<Host::SDFQuadraticSpline>& sdf, const BBox2f& sdfBounds, const ivec2& gridDims, AssetHandle<Host::DualImage3f>& inferenceImage) :
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


    __host__ int NNanoEvaluator::LoadTrainingSet(NNano::Cuda::Vector<InputSample>& inputSamples, NNano::Cuda::Vector<OutputSample>& targetSamples)
    {
        const auto numSamples = Area(m_gridDims);        
        inputSamples.Resize(numSamples);
        targetSamples.Resize(numSamples);
        
        auto [gridDims, blockDims] = Get1DLaunchParams(numSamples);
        KernelGenerateSDFDataset << <gridDims, blockDims, 0, m_cudaStream >> > (m_sdf->GetDeviceInstance(), inputSamples.GetComputeData(), targetSamples.GetComputeData(), m_sdfBounds, m_gridDims);
        IsOk(cudaStreamSynchronize(m_cudaStream));

        return numSamples;
    }

    __host__ int NNanoEvaluator::LoadInferenceBatch(NNano::Cuda::Vector<InputSample>& inputSamples, const int startIdx)
    {
        if (startIdx != 0) { return 0; }

        const auto numSamples = Area(m_gridDims);
        inputSamples.Resize(numSamples);

        auto [gridDims, blockDims] = Get1DLaunchParams(numSamples);
        KernelGenerateSDFDataset << <gridDims, blockDims, 0, m_cudaStream >> > (m_sdf->GetDeviceInstance(), inputSamples.GetComputeData(), nullptr, m_sdfBounds, m_gridDims);
        IsOk(cudaStreamSynchronize(m_cudaStream));
        
        return numSamples;
    }

    __host__ void NNanoEvaluator::StoreInferenceBatch(const NNano::Cuda::Vector<OutputSample>& outputSamples, const int startIdx)
    {
        Assert(outputSamples.Size() == Area(m_gridDims));

        auto [gridDims, blockDims] = Get1DLaunchParams(Area(m_gridDims));
        KernelRenderInference << <gridDims, blockDims, 0, m_cudaStream >> > (outputSamples.GetComputeData(), m_inferenceImage->GetDeviceInstance());
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

    __host__ void NNanoEvaluator::Run()
    {
        Log::Write("Running continuous NNano training...");

        m_mlp->PrepareTraining();

        while (m_threadState == kThreadRunning)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            Log::Debug("Tick...");
            m_mlp->Infer();
        }
    }
}