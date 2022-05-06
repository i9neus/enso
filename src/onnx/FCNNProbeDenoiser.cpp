#include "FCNNProbeDenoiser.h"
#include "generic/StringUtils.h"
#include "generic/HighResolutionTimer.h"
#include "kernels/math/CudaColourUtils.cuh"
#include <random>

#ifndef _DEBUG 

#include <onnxruntime/onnxruntime_cxx_api.h>

namespace ONNX
{
    FCNNProbeDenoiser::FCNNProbeDenoiser() :
        m_ortInputTensor(new Ort::Value(nullptr)),
        m_ortOutputTensor(new Ort::Value(nullptr)),
        m_isModelReady(false) {}

    FCNNProbeDenoiser::~FCNNProbeDenoiser() { Destroy(); }

    void FCNNProbeDenoiser::Initialise(const FCNNProbeDenoiserParams& params)
    {
        m_params = params;

        try
        {
            /*Destroy();

            m_ortEnvironment.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "grid2grid"));
            auto ortAllocatorInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

            m_inputTensorData.resize(3 * 8 * 8 * 32);
            m_outputTensorData.resize(3 * 8 * 8 * 32);
            m_inputTensorShape = { 1, 3, 8, 8, 32 };
            m_outputTensorShape = { 1, 3, 8, 8, 32 };

            *m_ortInputTensor = Ort::Value::CreateTensor<float>(ortAllocatorInfo, m_inputTensorData.data(), m_inputTensorData.size(),
                m_inputTensorShape.data(), m_inputTensorShape.size());
            *m_ortOutputTensor = Ort::Value::CreateTensor<float>(ortAllocatorInfo, m_outputTensorData.data(), m_outputTensorData.size(),
                m_outputTensorShape.data(), m_outputTensorShape.size());

            m_ortSession = std::make_unique< Ort::Session>(*m_ortEnvironment, Widen(modelPath).c_str(), Ort::SessionOptions(nullptr));

            m_isModelReady = true;
            m_modelPath = modelPath;*/
        }
        catch (const Ort::Exception& err)
        {
            AssertMsgFmt("Error: ONNX runtime: %s (code %i)", err.what(), err.GetOrtErrorCode());
        }

        m_isModelReady = true;
    }
    
    void FCNNProbeDenoiser::Reinitialise()
    {
        Initialise(m_params);
    }

    void FCNNProbeDenoiser::Destroy()
    {
        m_ortEnvironment.reset();
        m_ortSession.reset();
        m_isModelReady = false;
    }

    void FCNNProbeDenoiser::Evaluate(const std::vector<Cuda::vec3>& rawInputData, std::vector<Cuda::vec3>& rawOutputData)
    {
        if (!m_isModelReady) { return; }
        
        // Transpose the raw SH data into the tensor format required by the grid2grid model
        Timer timer;


        Log::SystemOnce("FCNNProbeDenoiser: Evaluated in %.2fms.", timer.Get() * 1e3f);
    }

    void FCNNProbeDenoiser::PackTensor(const std::vector<Cuda::vec3>& rawInputData, std::vector<float>& inputTensorData) const
    {
        //const std::vector<Cuda::vec3>& transformedInputData;
    }

    void FCNNProbeDenoiser::UnpackTensor(const std::vector<float>& outputTensorData, std::vector<Cuda::vec3>& rawOutputData) const
    {

    }
}

#endif 