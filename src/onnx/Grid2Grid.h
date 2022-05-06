#pragma once

#include "OnnxModel.h"

namespace ONNX
{
#ifdef DEBUG     
    class Grid2GridInterface : public ONNXModel
    {
    public:
        Grid2GridInterface() = default;
        Grid2GridInterface(const std::string& modelPath) {}

        void Reinitialise() {}
        void Initialise(const std::string& modelPath) {}
        void Evaluate(const std::vector<Cuda::vec3>& directData, std::vector<Cuda::vec3>& indirectData) {}
    };

#else
    class Grid2Grid
    {
    public:
        Grid2Grid();
        Grid2Grid(const std::string& modelPath);
        ~Grid2Grid();

        void Reinitialise();
        void Initialise(const std::string& modelPath);
        void Evaluate(const std::vector<Cuda::vec3>& directData, std::vector<Cuda::vec3>& indirectData);

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

        bool                            m_isModelReady;
        std::string                     m_modelPath;
    };

#endif
}