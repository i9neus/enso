#pragma once

#include "generic/StdIncludes.h"
#include "kernels/math/CudaMath.cuh"

// Forward declare a bunch of structures used by ONNX runtime
namespace Ort
{
    struct Env;
    struct Value;
    struct Session;
    struct SessionOptions;
}

namespace ONNX
{
    enum InferenceBackendType : int
    {
        kInferenceBackendCPU,
        kInferenceBackendCUDA,
        kInferenceBackendTensorRT
    };
    
    template<typename Type>
    struct Tensor
    {
        Tensor() = default;
        ~Tensor() = default;
        Tensor(Tensor&) = delete;
        Tensor(Tensor&& other) = delete;

         void Initialise(const std::vector<int64_t>& shape);
        std::string Format() const;

        std::vector<Type>               data;
        std::vector<int64_t>            shape;
    };

    template struct Tensor<float>;
    template struct Tensor<int64_t>;

    class OrtSessionParams
    {
    public:
        OrtSessionParams();
        ~OrtSessionParams();
        OrtSessionParams(const OrtSessionParams&) = delete;
        OrtSessionParams(const OrtSessionParams&&) = delete;

        void Destroy();
        void Initialise(const size_t numValues);
        void Finalise();
        template<typename Type> void CreateValue(const size_t idx, const std::string& name, Tensor<Type>& tensor);

        const char** GetValueNamesCstr() { return m_valueNamesCstr.data(); }
        inline Ort::Value* GetOrtValues() { return m_ortValues.data(); }
        int Size() const { return m_numValues; }

    private:
        std::vector<Ort::Value>                         m_ortValues;
        std::vector<std::string>                        m_valueNames;
        std::vector<const char*>                        m_valueNamesCstr;
        size_t                                          m_numValues;
    };

    template void OrtSessionParams::CreateValue<float>(const size_t, const std::string&, Tensor<float>&);
    template void OrtSessionParams::CreateValue<int64_t>(const size_t, const std::string&, Tensor<int64_t>&);
    
    class OrtSession
    {
    public:
        OrtSession() = default;
        ~OrtSession();

        void Destroy();
        void Initialise(Ort::Env& ortEnvironment, const Ort::SessionOptions& sessionOptions, const std::string& modelPath, const int numInputs, const int numOutputs);
        void Finalise();
        void Run();

        inline OrtSessionParams& Inputs() { return m_inputParams; }
        inline OrtSessionParams& Outputs() { return m_outputParams; }

    private:
        std::unique_ptr<Ort::Session>   m_ortSession;
        OrtSessionParams                m_inputParams;
        OrtSessionParams                m_outputParams;
    };    
    
    class ONNXModel
    {
    protected:
        ONNXModel() = default;
    };
}
