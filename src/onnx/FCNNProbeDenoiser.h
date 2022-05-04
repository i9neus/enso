#pragma once

#include "OnnxRuntime.h"

namespace ONNX
{    
    struct FCNNProbeDenoiserParams
    {
        std::string     modelRootPath;
        std::string     modelPreprocessPath;
        std::string     modelPostprocessPath; 
        std::string     modelDenoiserPath;

        Cuda::ivec3     gridResolution;
    };
    
#ifdef _DEBUG     
    class FCNNProbeDenoiser
    {
    public:
        FCNNProbeDenoiserInterface() = default;

        void Reinitialise() {}
        void Initialise(const FCNNProbeDenoiserParams& params) {}
        void Evaluate(const std::vector<Cuda::vec3>& inputData, std::vector<Cuda::vec3>& outputData) {}
    };

#else
    class FCNNProbeDenoiser
    {
    public:
        FCNNProbeDenoiser();
        ~FCNNProbeDenoiser();

        void Reinitialise();
        void Initialise(const FCNNProbeDenoiserParams& params);
        void Evaluate(const std::vector<Cuda::vec3>& inputData, std::vector<Cuda::vec3>& outputData);

    private:
        void Destroy();

        std::unique_ptr<Ort::Value>     m_ortInputTensor;
        std::unique_ptr<Ort::Value>     m_ortOutputTensor;
        std::unique_ptr<Ort::Session>   m_ortSession;
        std::unique_ptr<Ort::Env>       m_ortEnvironment;

        std::vector<float>              m_inputTensorData;
        std::vector<float>              m_outputTensorData;
        std::vector<int64_t>            m_inputTensorShape;
        std::vector<int64_t>            m_outputTensorShape;

        FCNNProbeDenoiserParams         m_params;
        bool                            m_isModelReady;
    };

#endif

}
