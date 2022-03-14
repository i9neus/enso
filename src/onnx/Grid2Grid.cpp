#include "Grid2Grid.h"
#include "generic/StringUtils.h"
#include <random>

#include <onnxruntime/onnxruntime_cxx_api.h>

namespace ONNX
{
    Grid2Grid::Grid2Grid() :
        m_ortInputTensor(new Ort::Value(nullptr)),
        m_ortOutputTensor(new Ort::Value(nullptr)),
        m_isModelReady(false){}

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

            m_isModelReady = true;
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
        m_isModelReady = false;
    }

    void Grid2Grid::Evaluate(const std::vector<Cuda::vec3>& rawInputData, std::vector<Cuda::vec3>& rawOutputData)
    {
        if (!m_isModelReady) { return; }
        
        /*std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<> rng(0.0f, 1.0f);*/

        static const int shSwizzle[4] = {0, 3, 1, 2};
        static const float shSign[4] = {1.0, -1.0, -1.0, 1.0};
        
        // Transpose the raw SH data into the tensor format required by the grid2grid model
        for (int c = 0, tensorIdx = 0; c < 3; ++c)
        {
            for(int z = 0; z < 8; ++z)
            {
                for (int y = 0; y < 8; ++y)
                {
                    for (int x = 0; x < 8; ++x, tensorIdx += 4)
                    {
                        const int inputIdx = (64 * (7 - z) + 8 * y + x) * 5;
                        for (int i = 0; i < 4; ++i)
                        {
                            float sh = shSign[i] * rawInputData[inputIdx][c + shSwizzle[i]];
                            sh = std::copysign(std::log(1.0 + std::abs(sh)), sh);
                            m_inputTensorData[tensorIdx + i] = sh;
                        }
                    }
                }
            }
        }
        
        // Evaluate the model
        try
        {
            static const char* inputNames[] = { "input" };
            static const char* outputNames[] = { "output" };

            m_ortSession->Run(Ort::RunOptions(nullptr), inputNames, m_ortInputTensor.get(), 1, outputNames, m_ortOutputTensor.get(), 1);

            Log::SystemOnce("grid2grid: Okay!");
        }
        catch (const Ort::Exception& err)
        {
            AssertMsgFmt("Error: ONNX runtime: %s (code %i)", err.what(), err.GetOrtErrorCode());
        }

        // Transpose the data back again 
        for (int c = 0, tensorIdx = 0; c < 3; ++c)
        {
            for (int z = 0; z < 8; ++z)
            {
                for (int y = 0; y < 8; ++y)
                {
                    for (int x = 0; x < 8; ++x, tensorIdx += 4)
                    {
                        const int outputIdx = (64 * (7 - z) + 8 * y + x) * 5;
                        for (int i = 0; i < 4; ++i)
                        {
                            float sh = m_outputTensorData[tensorIdx + i];
                            sh = std::copysign(std::exp(std::abs(sh)) - 1.0f, sh);
                            rawOutputData[outputIdx + shSwizzle[i]] = shSign[i] * sh;
                        }
                    }
                }
            }
        }
    }
}