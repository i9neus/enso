#include "Grid2Grid.h"
#include "generic/StringUtils.h"

#include <onnxruntime/onnxruntime_cxx_api.h>

namespace ONNX
{
    Grid2Grid::Grid2Grid() :
        m_ortInputTensor(new Ort::Value(nullptr)),
        m_ortOutputTensor(new Ort::Value(nullptr)),
        m_signalIsOkay(false) {}

    Grid2Grid::Grid2Grid(const std::string& modelPath) : Grid2Grid()       
    {
        Initialise(modelPath);
    }

    Grid2Grid::~Grid2Grid() { Destroy(); }

    void Grid2Grid::Initialise(const std::string& modelPath)
    {
        try
        {
            Destroy();
            
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
        }
        catch (const Ort::Exception& err)
        {
            AssertMsgFmt("Error: ONNX runtime: %s (code %i)", err.what(), err.GetOrtErrorCode());
        }
    }

    void Grid2Grid::Destroy()
    {
        m_ortEnvironment.reset();
        m_ortSession.reset();
    }

    void Grid2Grid::Evaluate(const std::vector<Cuda::vec3>& rawInputData, std::vector<Cuda::vec3>& rawOutputData)
    {
        // Transpose the raw SH data into the tensor format required by the grid2grid model
        for (int c = 0, idx = 0; c < 3; ++c)
        {
            for (int p = 0; p < 8 * 8 * 8; ++p)
            {
                for (int sh = 0; sh < 4; ++sh, ++idx)
                {
                    m_inputTensorData[idx] = rawInputData[p * 5 + sh][c];
                }
            }
        }
        
        // Evaluate the model
        try
        {
            static const char* inputNames[] = { "input" };
            static const char* outputNames[] = { "output" };

            m_ortSession->Run(Ort::RunOptions(nullptr), inputNames, m_ortInputTensor.get(), 1, outputNames, m_ortOutputTensor.get(), 1);

            Log::System("grid2grid: Okay!");            
        }
        catch (const Ort::Exception& err)
        {
            AssertMsgFmt("Error: ONNX runtime: %s (code %i)", err.what(), err.GetOrtErrorCode());
        }

        // Transpose the data back again 
        for (int c = 0, idx = 0; c < 3; ++c)
        {
            for (int p = 0; p < 8 * 8 * 8; ++p)
            {
                for (int sh = 0; sh < 4; ++sh, ++idx)
                {
                    rawOutputData[p * 5 + sh][c] = m_outputTensorData[idx];
                }
            }
        }
    }
}