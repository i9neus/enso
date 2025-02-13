#include "NNanoEvaluator.cuh"
#include "core/utils/HighResolutionTimer.h"
#include "core/math/samplers/PCG.cuh"
#include "core/math/pdf/PDF2.cuh"
#include "core/containers/DualImageOps.cuh"

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
        CudaAssertDebug(inputData);

        const auto kernelIdx = kKernelIdx;
        if (kernelIdx < Area(setDims))
        {
            const vec2 p = TransformNormalisedScreenToView(vec2(float(kernelIdx % setDims.x) / float(setDims.x), float(kernelIdx / setDims.x) / float(setDims.y)), setDims);
            
            inputData[kernelIdx] = PositionalEncode(p);
            //inputData[kernelIdx] = InputSample({ p.x, p.y });
        }
    }
    
    __global__ void KernelGenerateTrainingSet(const Device::SDFQuadraticSpline* sdf, const Device::PDF2* pdf, InputSample* inputData, OutputSample* targetData, const BBox2f sdfBounds, const int numSamples, const int seed)
    {
        CudaAssertDebug(sdf);
        CudaAssertDebug(inputData || targetData);

        const auto kernelIdx = kKernelIdx;
        if (kernelIdx < numSamples)
        {
            // Generate random numbers in the range [0, 1]
            const vec2 xi = PCG(HashOf(kernelIdx, seed)).Rand().xy;

            vec2 q;
            vec3 f = pdf->Sample(xi, q);                  
            //vec2 q = xi;
            //vec3 f = pdf->Evaluate(xi);

            const vec2 p = TransformNormalisedScreenToView(q, sdfBounds.Dimensions());

            inputData[kernelIdx] = PositionalEncode(p);
            //inputData[kernelIdx] = InputSample({ p.x, p.y });
            targetData[kernelIdx] = NNano::Activation::TanH::F(f.z);
        }
    }

    template<typename SampleType>
    __global__ void KernelRenderTensorField(const SampleType* targetData, Device::DualImage3f* image, const Device::PDF2* pdf, const Device::SDFQuadraticSpline* sdf, const int offset, const int frameIdx)
    {
        CudaAssertDebug(targetData);
        CudaAssertDebug(image);

        const int kernelIdx = kKernelIdx, width = image->Width(), height = image->Height();
        if (kernelIdx < image->Area())
        {
            const int x = kernelIdx % width, y = kernelIdx / width;
            vec3& outPixel = image->As<vec3>(x, y);
            
            // Evaluate the value of the SDF
            float f = targetData[kernelIdx][offset];
            f = NNano::Activation::TanH::InvF(f);
            f = cosf(10.f * kTwoPi * f) * 0.5f + 0.5f;

            ivec2 dims = pdf->Dims();
            const vec2 p = TransformNormalisedScreenToView(vec2(float(x) / width, float(y) / height), dims);
            outPixel.xy = normalize(sdf->Evaluate(p).yz - p);
            outPixel.z = 0.;
            outPixel = outPixel * 0.5 + 0.5;

            const mat2 H = sdf->Hessian(p, 50. / pdf->Height());
            outPixel.xy = H[0];

            // Write to the buffer

            //outPixel = f;
            if (pdf && x < pdf->Width() && y < pdf->Height())
            {
                const vec3 f = pdf->Evaluate(vec2(x / float(pdf->Width()), y / float(pdf->Height())));
                outPixel = f.z;
                //outPixel[2] = cdf->At(x, y);
                
                //outPixel = 0.;

                PCG rng(HashOf(x, y, frameIdx));
                for (int i = 0; i < 100; ++i)
                {
                    vec2 xi = rng.Rand().xy;
                    vec2 p;
                    pdf->Sample(xi, p);

                    if (length(p - vec2(float(x) / pdf->Width(), float(y) / pdf->Height())) < 0.01)
                        outPixel = vec3(1., 0., 0.);
                }
            }
        }
    }

    __global__ void KernelPopulatePDF(const Device::SDFQuadraticSpline* sdf, Device::PDF2* pdf)
    {
        CudaAssertDebug(sdf && pdf);   

        const auto kernelIdx = kKernelIdx;
        if (kernelIdx < pdf->Area())
        {
            const int width = pdf->Width();
            const int x = kernelIdx % width, y = kernelIdx / width;
            const vec2 p = TransformNormalisedScreenToView(vec2(float(x) / width, float(y) / pdf->Height()), pdf->Dims());

            // Evaluate the PDF
            const float f = sdf->Evaluate(p)[0];

            // Calculate the determinant of the Hessian matrix at p, then map it to the range [0, 1]
            const mat2 H = sdf->Hessian(p, 50. / pdf->Height());
            const float detH = mix(1e-3f, 1.f, NNano::Activation::TanH::F(fabsf(det(H))));
            //const float detH = 1.;
            //const float detH = mix(1e-3f, 1.f, saturatef(1 - f));

            // Write to the buffer
            pdf->Set(x, y, detH, detH);
        }
    }
    
    __host__ NNanoEvaluator::NNanoEvaluator(const InitCtx& initCtx, AssetHandle<Host::SDFQuadraticSpline>& sdf, const BBox2f& sdfBounds, const ivec2& infGridDims, AssetHandle<Host::DualImage3f>& inferenceImage) :
        Host::Dirtyable(initCtx),
        m_threadState(kThreadRunning),
        m_sdf(sdf),
        m_sdfBounds(sdfBounds),
        m_gridDims(infGridDims),
        m_inferenceImage(inferenceImage)
    {
        // Create a new stream that will run all NNano operations for this model
        IsOk(cudaStreamCreate(&m_cudaStream));

        // Create the model
        m_mlp.reset(new MLP(*this, m_cudaStream));

        // Create a PDF we can sample from
        m_hostPDF = AssetAllocator::CreateChildAsset<Host::PDF2>(*this, "pdf", 512, 256, m_cudaStream);
        PreparePDF();

        // Launch the worker thread
        m_workerThread = std::thread(&NNanoEvaluator::Run, this);
    }

    __host__ void NNanoEvaluator::PreparePDF()
    {        
        Log::Error("Area: %i", m_hostPDF->Area());
        auto [gridDims, blockDims] = Get1DLaunchParams(m_hostPDF->Area());
        KernelPopulatePDF << <gridDims, blockDims, 0, m_cudaStream >> > (m_sdf->GetDeviceInstance(), m_hostPDF->GetDeviceInstance());
        IsOk(cudaStreamSynchronize(m_cudaStream));

        Dilate(m_hostPDF->GetCDF(), kImageDilateSquare, 1, m_cudaStream, kImageRMask);

        m_hostPDF->Rebuild();            
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
        auto& cdf = m_hostPDF->GetCDF();

        auto [gridDims, blockDims] = Get1DLaunchParams(Area(m_gridDims));
        KernelRenderTensorField<< <gridDims, blockDims, 0, m_cudaStream >> > (outputSamples.GetComputeData(), m_inferenceImage->GetDeviceInstance(), m_hostPDF->GetDeviceInstance(), m_sdf->GetDeviceInstance(), 0, m_epochIdx);
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

            m_hostPDF.DestroyAsset();
        }
    }

    __host__ void NNanoEvaluator::PrepareEpoch(const int epochIdx)
    {
        auto [inputSamples, targetSamples] = m_mlp->GetTrainingDataObjects();
        
        const int kNumSamples = MLP::kMiniBatchSize * 50;
        inputSamples->Resize(kNumSamples);
        targetSamples->Resize(kNumSamples);

        auto [gridDims, blockDims] = Get1DLaunchParams(kNumSamples, 256);
        KernelGenerateTrainingSet << <gridDims, blockDims, 0, m_cudaStream >> > (m_sdf->GetDeviceInstance(), m_hostPDF->GetDeviceInstance(), inputSamples->GetComputeData(), targetSamples->GetComputeData(), m_sdfBounds, kNumSamples, epochIdx);
        IsOk(cudaStreamSynchronize(m_cudaStream));

        m_mlp->PrepareTraining();
    }

    __host__ void NNanoEvaluator::Run()
    {
        Log::Write("Running continuous NNano training...");

        m_mlp->Initialise();
        
        HighResolutionTimer inferenceTimer;
        for(m_epochIdx = 0; m_threadState == kThreadRunning; ++m_epochIdx)
        {
            PrepareEpoch(m_epochIdx);
            
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