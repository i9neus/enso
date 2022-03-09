#pragma once

#include "generic/StdIncludes.h"

// Forward declare a bunch of structures used by ONNX runtime
namespace Ort
{
    struct Env;
    struct Value;
    struct Session;
}

namespace ONNX
{
    class Grid2Grid
    {
    public:
        Grid2Grid();
        Grid2Grid(const std::string& modelPath);
        ~Grid2Grid();

        void Initialise(const std::string& modelPath);
        void Evaluate(const std::vector<float>& directData, std::vector<float>& indirectData);

    private:       
        void RunSession(const std::vector<float>& directData, std::vector<float>& indirectData);
        void Destroy();

        std::unique_ptr<Ort::Value>     m_ortInputTensor;
        std::unique_ptr<Ort::Value>     m_ortOutputTensor;
        std::unique_ptr<Ort::Session>   m_ortSession;
        std::unique_ptr<Ort::Env>       m_ortEnvironment;

        std::vector<float>              m_inputTensorData;
        std::vector<float>              m_outputTensorData;
        std::vector<int64_t>            m_inputTensorShape;
        std::vector<int64_t>            m_outputTensorShape;

    };
}