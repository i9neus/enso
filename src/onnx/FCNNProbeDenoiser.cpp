#include "FCNNProbeDenoiser.h"
#include "generic/StringUtils.h"
#include "generic/HighResolutionTimer.h"
#include "kernels/math/CudaColourUtils.cuh"
#include <random>

#include "generic/JsonUtils.h"
#include "generic/FilesystemUtils.h"

#ifndef _DEBUG 
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

namespace ONNX
{

    FCNNProbeDenoiserParams::FCNNProbeDenoiserParams()
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

        try
        {
            Destroy();

            m_ortEnvironment.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "fcnndenoiser"));            

            // Create session objects
            m_ortPreprocSession = std::make_unique<Ort::Session>(*m_ortEnvironment, Widen(modelPreprocessPath).c_str(), Ort::SessionOptions(nullptr));
            m_ortDenoiserSession = std::make_unique<Ort::Session>(*m_ortEnvironment, Widen(modelDenoiserPath).c_str(), Ort::SessionOptions(nullptr));
            m_ortPostprocSession = std::make_unique<Ort::Session>(*m_ortEnvironment, Widen(modelPostprocessPath).c_str(), Ort::SessionOptions(nullptr));

            m_initFlags |= kInitORT;
            Log::Success("Successfully initialised PCNN denoiser models");
        }
        catch (const Ort::Exception& err)
        {
            Log::Error("ONNX runtime error %i: '%s'", err.GetOrtErrorCode(), err.what());
        }
    }

    void FCNNProbeDenoiser::Reinitialise()
    {
        Initialise(m_params);
    }

    void FCNNProbeDenoiser::Destroy()
    {
        m_ortEnvironment.reset();
        m_ortPreprocSession.reset();
        m_ortDenoiserSession.reset();
        m_ortPostprocSession.reset();

        m_unprocSHTensor.Destroy();
        m_procNoisySHTensor.Destroy();
        m_procDenoisedSHTensor.Destroy();
        m_unprocMaskTensor.Destroy();
        m_procMaskTensor.Destroy();
        m_padTensor.Destroy();
        m_statTensor.Destroy();

        m_initFlags = 0;
    }

    void FCNNProbeDenoiser::Evaluate(const FCNNProbeDenoiserParams& evalParams, std::vector<Cuda::vec3>& rawInputData, std::vector<Cuda::vec3>& rawOutputData)
    {
        if (!(m_initFlags | kInitORT))
        {
            Log::Error("Error: ONNX runtime is not initialised: %u", m_initFlags);
            return;
        }

        Timer timer;
        const int w = evalParams.grid.gridDensity.x, h = evalParams.grid.gridDensity.y, d = evalParams.grid.gridDensity.z;

        if (evalParams.grid.gridDensity != m_params.grid.gridDensity || !(m_initFlags | kInitTensors))
        {
            // Initialise the tensor wrapper objects
            m_unprocSHTensor.Initialise({ 1, h, w, d, 12 });
            m_procNoisySHTensor.Initialise({ 1, 12, h, w, d });
            m_procDenoisedSHTensor.Initialise({ 1, 12, h, w, d });

            m_unprocMaskTensor.Initialise({ 1, h, w, d, 1 });
            m_procMaskTensor.Initialise({ 1, 1, h, w, d });

            m_padTensor.Initialise({ 6 });
            m_statTensor.Initialise({ 2 });

            m_packedCoeffData.reserve(w * d * h * 5);

            m_initFlags |= kInitTensors;
            Log::Debug("Reinitialised tensors.");
        }
        if (evalParams.grid.dataTransform != m_params.grid.dataTransform  || !(m_initFlags | kInitTransforms))
        {
            // Constuct a transform for the grid
            m_dataTransform.Construct(evalParams.grid);

            m_initFlags |= kInitTransforms;
            Log::Debug("Reconstructed probe data transform");
        }        
        m_params = evalParams;       

        m_params.grid.Echo();

        // Evaluate the model
        try
        {
            // Transform the probe data into the space used by the model
            m_dataTransform.Forward(rawInputData, m_packedCoeffData);
            
            // Unpack the into the input tensors
            m_dataTransform.UnpackCoefficients(m_packedCoeffData, m_unprocSHTensor.data, m_unprocMaskTensor.data);
            
            // Execute the pre-process model
            static const char* preProcInputNames[2] = { "input", "mask" };
            static const char* preProcOutputNames[4] = { "output", "mask_in", "pad", "stat" };
            Ort::Value* const preProcInputPtrs[2] = { m_unprocSHTensor.ort.get(), m_unprocMaskTensor.ort.get() };
            Ort::Value* const preProcOutputPtrs[4] = { m_procNoisySHTensor.ort.get(), m_procMaskTensor.ort.get(), m_padTensor.ort.get(), m_statTensor.ort.get() };
            m_ortPreprocSession->Run(Ort::RunOptions(nullptr), preProcInputNames, preProcInputPtrs[0], 2, preProcOutputNames, preProcOutputPtrs[0], 4);

            // Execute the denoise model
            static const char* denoiseInputNames[2] = { "input", "mask" };
            static const char* denoiseOutputNames[1] = { "output" };
            Ort::Value* const denoiseInputPtrs[2] = { m_procNoisySHTensor.ort.get(), m_procMaskTensor.ort.get() };
            Ort::Value* const denoiseOutputPtrs[1] = { m_procDenoisedSHTensor.ort.get() };
            m_ortPreprocSession->Run(Ort::RunOptions(nullptr), denoiseInputNames, denoiseInputPtrs[0], 2, denoiseOutputNames, denoiseOutputPtrs[0], 1);

            // Execute the post-process model
            static const char* postProcInputNames[3] = { "input", "pad", "stat" };
            static const char* postProcOutputNames[1] = { "output" };
            Ort::Value* const postProcInputPtrs[3] = { m_procNoisySHTensor.ort.get(), m_padTensor.ort.get(), m_statTensor.ort.get() };
            Ort::Value* const postProcOutputPtrs[1] = { m_unprocSHTensor.ort.get() };
            m_ortPreprocSession->Run(Ort::RunOptions(nullptr), postProcInputNames, postProcInputPtrs[0], 3, postProcOutputNames, postProcOutputPtrs[0], 1);

            // Pack the denoised data 
            m_dataTransform.PackCoefficients(m_unprocSHTensor.data, m_unprocMaskTensor.data, m_packedCoeffData);
            
            // Invert the data transform
            m_dataTransform.Inverse(m_packedCoeffData, rawOutputData);

            Log::SystemOnce("grid2grid: Okay!");
        }
        catch (const Ort::Exception& err)
        {
            Log::Error("Error: ONNX runtime: %s (code %i)", err.what(), err.GetOrtErrorCode());
        }

        Log::SystemOnce("FCNNProbeDenoiser: Evaluated in %.2fms.", timer.Get() * 1e3f);
    }

#endif 

}