#include "FCNNProbeDenoiser.h"
#include "generic/StringUtils.h"
#include "generic/HighResolutionTimer.h"
#include "kernels/math/CudaColourUtils.cuh"
#include <random>

#include "generic/JsonUtils.h"
#include "generic/FilesystemUtils.h"

#ifndef _DEBUG 
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/tensorrt_provider_factory.h>
#endif

namespace ONNX
{
    FCNNProbeDenoiserParams::FCNNProbeDenoiserParams() : 
        inferenceBackend(kInferenceBackendCPU)
    {
        grid.gridDensity = Cuda::ivec3(-1);
    }

#ifndef _DEBUG 

    FCNNProbeDenoiser::FCNNProbeDenoiser() :
        m_initFlags(0u) {}

    FCNNProbeDenoiser::~FCNNProbeDenoiser() { Destroy(); }

    void FCNNProbeDenoiser::Initialise(const FCNNProbeDenoiserParams& params)
    {
        bool filesOkay = true;
        auto ProcessPath = [&params, &filesOkay](const std::string& inPath) -> std::string
        {
            std::string outPath = params.modelRootPath.empty() ? 
                                  inPath :
                                  (SlashifyPath(params.modelRootPath) + inPath);
            
            if (!FileExists(outPath))
            {
                Log::Error("Error: ONNX model '%s' not found.", outPath);
            }
            return outPath;
        };
        
        const std::string modelPreprocessPath = ProcessPath(params.modelPreprocessPath);
        const std::string modelDenoiserPath = ProcessPath(params.modelDenoiserPath);
        const std::string modelPostprocessPath = ProcessPath(params.modelPostprocessPath);

        if (!filesOkay) { return; }
        m_params = params;

        try
        {
            Destroy();

            m_ortEnvironment.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "fcnndenoiser"));

            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads(1);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            // Select the backend
            if (m_params.inferenceBackend == kInferenceBackendCUDA)
            {
                /*OrtCUDAProviderOptions options;
                options.device_id = 0;
                options.arena_extend_strategy = 0;
                options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
                options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
                options.do_copy_in_default_stream = 1;*/

                Log::Warning("Using CUDA backend");

                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
            }
            else if (m_params.inferenceBackend == kInferenceBackendTensorRT)
            {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));

                Log::Warning("Using TensorRT backend");
            } 
            else
            {
                Log::Warning("Using CPU backend");
            }

            // Create session objects
            m_preProcSession.Initialise(*m_ortEnvironment, sessionOptions, modelPreprocessPath, 2, 4);
            m_denoiseSession.Initialise(*m_ortEnvironment, sessionOptions, modelDenoiserPath, 2, 1);
            m_postProcSession.Initialise(*m_ortEnvironment, sessionOptions, modelPostprocessPath, 3, 1);

            m_initFlags |= kInitORT;
            Log::Success("Successfully initialised FCNN denoiser models");
        }
        catch (const Ort::Exception& err)
        {
            Log::Error("ONNX runtime error %i: '%s'", err.GetOrtErrorCode(), err.what());
        }
    }

    void FCNNProbeDenoiser::Reinitialise()
    {
        Log::Warning("Re-initialising FCNNProbeDenoiser model...");

        Initialise(m_params);
    }

    void FCNNProbeDenoiser::Destroy()
    {
        try
        {
            m_preProcSession.Destroy();
            m_denoiseSession.Destroy();
            m_postProcSession.Destroy();

            m_ortEnvironment.reset();
        }
        catch (const Ort::Exception& err)
        {
            Log::Error("ONNX runtime error %i: '%s'", err.GetOrtErrorCode(), err.what());
        }

        m_initFlags = 0;
    }

    void FCNNProbeDenoiser::Evaluate(const FCNNProbeDenoiserParams& evalParams, const std::vector<Cuda::vec3>& rawInputData, std::vector<Cuda::vec3>& rawOutputData)
    {
        if (!(m_initFlags & kInitORT))
        {
            Log::Error("Error: ONNX runtime is not initialised: %u", m_initFlags);
            return;
        }

        Timer timer;       

        if (evalParams.grid.gridDensity != m_params.grid.gridDensity || !(m_initFlags & kInitTensors))
        {  
            const int w = evalParams.grid.gridDensity.x, h = evalParams.grid.gridDensity.y, d = evalParams.grid.gridDensity.z;
            
            // Initialise the tensor wrapper objects
            m_unprocSHTensor.Initialise({ 1, h, w, d, 12 });
            m_procNoisySHTensor.Initialise({ 1, 14, h, w, d });
            m_procDenoisedSHTensor.Initialise({ 1, 13, h, w, d });

            m_unprocMaskTensor.Initialise({ 1, h, w, d, 1 });
            m_procMaskTensor.Initialise({ 1, 1, h, w, d });

            m_padTensor.Initialise({ 6 });
            m_statTensor.Initialise({ 2 });

            // Create the pre-process ORT session
            m_preProcSession.Inputs().CreateValue(0, "input", m_unprocSHTensor);
            m_preProcSession.Inputs().CreateValue(1, "mask", m_unprocMaskTensor);
            m_preProcSession.Outputs().CreateValue(0, "output", m_procNoisySHTensor);
            m_preProcSession.Outputs().CreateValue(1, "mask_in", m_procMaskTensor);
            m_preProcSession.Outputs().CreateValue(2, "pad", m_padTensor);
            m_preProcSession.Outputs().CreateValue(3, "stat", m_statTensor);
            m_preProcSession.Finalise();

            // Create the denoiser ORT session
            m_denoiseSession.Inputs().CreateValue(0, "input", m_procNoisySHTensor);
            m_denoiseSession.Inputs().CreateValue(1, "mask", m_procMaskTensor);
            m_denoiseSession.Outputs().CreateValue(0, "output", m_procDenoisedSHTensor);
            m_denoiseSession.Finalise();           

            // Create the post-process ORT session
            m_postProcSession.Inputs().CreateValue(0, "input", m_procDenoisedSHTensor);
            m_postProcSession.Inputs().CreateValue(1, "pad", m_padTensor);
            m_postProcSession.Inputs().CreateValue(2, "stat", m_statTensor);
            m_postProcSession.Outputs().CreateValue(0, "output", m_unprocSHTensor);
            m_postProcSession.Finalise();

            m_initFlags |= kInitTensors;
            Log::Warning("Reinitialised tensors.");
        }
        if (evalParams.grid.dataTransform != m_params.grid.dataTransform  || !(m_initFlags & kInitTransforms))
        {
            // Constuct a transform for the grid
            m_dataTransform.Construct(evalParams.grid);

            m_initFlags |= kInitTransforms;
            Log::Warning("Reconstructed probe data transform");
        }     
        
        m_params = evalParams;       
        Assert(m_initFlags == kIsModelReady);

        // Evaluate the model
        try
        {
            Log::Indent indent("FCNNProbeDenoiser: Running...");
            
            // Transform the probe data into the space used by the model
            m_dataTransform.Forward(rawInputData, m_packedCoeffData);
            
            // Unpack the data into the input tensors
            m_dataTransform.UnpackCoefficients(m_packedCoeffData, m_unprocSHTensor.data, m_unprocMaskTensor.data);
            
            // Execute the pre-process model
            m_preProcSession.Run();
            Log::WriteOnce("FCNNProbeDenoiser: Pre-process in %.2fms.", timer.Get() * 1e3f);

            /*for (auto i : m_padTensor.data) Log::Warning("%i", i);
            for (auto i : m_statTensor.data) Log::Warning("%f", i);
            for (int i = 0; i < 100 && i < m_procNoisySHTensor.data.size(); ++i)
            {
                Log::Warning("%f, %f", m_procNoisySHTensor.data[i], m_procMaskTensor.data[i]);
            }*/

            m_denoiseSession.Run();
            Log::Write("FCNNProbeDenoiser: Denoiser in %.2fms.", timer.Get() * 1e3f);

            m_postProcSession.Run();
            Log::Write("FCNNProbeDenoiser: Post-process in %.2fms.", timer.Get() * 1e3f);
          
            // Pack the denoised data 
            m_dataTransform.PackCoefficients(m_unprocSHTensor.data, m_unprocMaskTensor.data, m_packedCoeffData);
            
            // Invert the data transform
            m_dataTransform.Inverse(m_packedCoeffData, rawOutputData);
        }
        catch (const Ort::Exception& err)
        {
            Log::Error("Error: ONNX runtime: %s (code %i)", err.what(), err.GetOrtErrorCode());
        }
        catch (...)
        {
            Log::Error("Error: ONNX runtime: unhandled error.");
        }

        Log::Success("FCNNProbeDenoiser: Evaluated in %.2fms.", timer.Get() * 1e3f);
    }

#endif 

}