#include "Grid2Grid.h"
#include "generic/StringUtils.h"
#include "generic/HighResolutionTimer.h"
#include "kernels/math/CudaColourUtils.cuh"
#include <random>

#ifndef _DEBUG

#include <onnxruntime/onnxruntime_cxx_api.h>

namespace ONNX
{
    Grid2Grid::Grid2Grid() :
        m_isModelReady(false)
    {
        m_ortInputTensor = std::make_unique<Ort::Value>(nullptr);
        m_ortOutputTensor = std::make_unique<Ort::Value>(nullptr);
    }

    Grid2Grid::Grid2Grid(const std::string& modelPath) : Grid2Grid()       
    {
        Initialise(modelPath);
    }

    Grid2Grid::~Grid2Grid() { Destroy(); }

    void Grid2Grid::Reinitialise()
    {
        Initialise(m_modelPath);
    }

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
            m_modelPath = modelPath;
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
        static const float shSign[4] = {1.0, 1.0, 1.0, -1.0};

        constexpr int W = 8, H = 8, D = 8;
        constexpr int kNumProbes = W * H * D;
        constexpr int kTensorChannelStride = kNumProbes * 4;

        Assert(rawInputData.size() >= kNumProbes * 5);
        Assert(rawOutputData.size() >= kNumProbes * 5);
        
        // Transpose the raw SH data into the tensor format required by the grid2grid model
        Timer timer;
        for (int z = 0, inputIdx = 0; z < 8; ++z)
        {
            for (int y = 0; y < 8; ++y)
            {
                for (int x = 0; x < 8; ++x, inputIdx+=5)
                {
                    const int tensorIdx = (W * H * (W - 1 - z) + W * y + x) * 4;
                    for (int i = 0; i < 4; ++i)
                    {
                        auto sh = rawInputData[inputIdx + shSwizzle[i]] * shSign[i];
                        //sh = log(abs(sh) + 1.0f) * sign(sh);
 
                        m_inputTensorData[tensorIdx + i] = sh[0];
                        m_inputTensorData[tensorIdx + i + kTensorChannelStride] = sh[1];
                        m_inputTensorData[tensorIdx + i + kTensorChannelStride * 2] = sh[2];
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
        for (int z = 0, outputIdx = 0; z < 8; ++z)
        {
            for (int y = 0; y < 8; ++y)
            {
                for (int x = 0; x < 8; ++x, outputIdx+=5)
                {
                    const int tensorIdx = (W*H * (W-1-z) + W*y + x) * 4;
                    for (int i = 0; i < 4; ++i)
                    {
                        Cuda::vec3 sh(m_outputTensorData[tensorIdx + i],
                                      m_outputTensorData[tensorIdx + i + kTensorChannelStride], 
                                      m_outputTensorData[tensorIdx + i + kTensorChannelStride*2]);
                        
                        //sh = (exp(abs(sh)) - 1.0f) * sign(sh);                      
                        rawOutputData[outputIdx + shSwizzle[i]] = sh * shSign[i];
                    }
                }
            }
        }

        Log::SystemOnce("grid2grid: Evaluated in %.2fms.", timer.Get() * 1e3f);
    }
}

#endif 