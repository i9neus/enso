#include "OnnxModel.h"
#include "generic/StringUtils.h"

#ifndef _DEBUG

#include <onnxruntime/onnxruntime_cxx_api.h>

namespace ONNX
{
    template<typename Type>
    void Tensor<Type>::Initialise(const std::vector<int64_t>& sh)
    {
        // Allocate host memory
        int numElements = 1;
        bool isValid = true;
        shape = sh;
        for (int idx = 0; idx < shape.size(); ++idx) 
        { 
            if (shape[idx] > 0) { numElements *= shape[idx]; }
            else                { isValid = false; }
        }

        AssertMsgFmt(isValid, "Tensor shape %s is illegal.", Format().c_str());        
        data.resize(numElements);   
    }

    template<typename Type>
    std::string Tensor<Type>::Format() const
    {
        std::string shapeStr = "{ ";
        for (int idx = 0; idx < shape.size(); ++idx)
        {
            shapeStr += tfm::format("%s%i", (idx == 0) ? "" : ", ", shape[idx]);
        }
        return shapeStr + " }";
    }

    OrtSessionParams::OrtSessionParams() : 
        m_numValues(0)
    {
    }    

    OrtSessionParams::~OrtSessionParams()
    {
        Destroy();
    }

    void OrtSessionParams::Destroy()
    {
        for (auto& value : m_ortValues) 
        { 
            value.release(); 
        }
        m_ortValues.clear();
    }

    void OrtSessionParams::Initialise(const size_t numValues)
    {
        Assert(numValues != 0);

        Destroy();

        m_numValues = numValues;
        m_valueNames.resize(m_numValues);
        m_valueNamesCstr.resize(m_numValues);
        for (int i = 0; i < m_numValues; ++i)
        {
            m_ortValues.emplace_back(nullptr);            
        }
    }

    void OrtSessionParams::Finalise()
    {
        for (int idx = 0; idx < m_valueNames.size(); ++idx)
        {
            m_valueNamesCstr[idx] = m_valueNames[idx].c_str();
        }
    }

    template<typename Type> void OrtSessionParams::CreateValue(const size_t idx, const std::string& name, Tensor<Type>& tensor)
    {
        AssertMsgFmt(idx >= 0 && idx < m_ortValues.size(), "Index %i is out of bounds [0, %i)", idx, m_ortValues.size());
        Assert(!name.empty());

        m_valueNames[idx] = name;    

        // Allocate the ONNX tensor
        auto ortAllocatorInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        m_ortValues[idx].release();
        m_ortValues[idx] = Ort::Value::CreateTensor<Type>(ortAllocatorInfo, tensor.data.data(), tensor.data.size(), tensor.shape.data(), tensor.shape.size());

        Log::Warning("Created ORT tensor with shape %s  (%i elements).", tensor.Format(), tensor.shape.size());
    }

    OrtSession::~OrtSession()
    {
        Destroy();
    }

    void OrtSession::Destroy()
    {
        m_ortSession.reset();
    }

    void OrtSession::Initialise(Ort::Env& ortEnvironment, const Ort::SessionOptions& sessionOptions, const std::string& modelPath, const int numInputs, const int numOutputs)
    {
        m_ortSession = std::make_unique<Ort::Session>(ortEnvironment, Widen(modelPath).c_str(), sessionOptions);

        m_inputParams.Initialise(numInputs);
        m_outputParams.Initialise(numOutputs);
    } 

    void OrtSession::Finalise()
    {
        m_inputParams.Finalise();
        m_outputParams.Finalise();
    }

    void OrtSession::Run()
    {
        m_ortSession->Run(Ort::RunOptions(nullptr), m_inputParams.GetValueNamesCstr(), m_inputParams.GetOrtValues(), m_inputParams.Size(), 
                                                    m_outputParams.GetValueNamesCstr(), m_outputParams.GetOrtValues(), m_outputParams.Size());
    }
}

#endif