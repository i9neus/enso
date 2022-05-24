#pragma once

#include "OnnxModel.h"
#include "kernels/lightprobes/CudaLightProbeGrid.cuh"
#include "kernels/lightprobes/CudaLightProbeDataTransform.cuh"

namespace ONNX
{    
    struct FCNNProbeDenoiserParams
    {
        __host__ __device__ FCNNProbeDenoiserParams();

        std::string     modelRootPath;
        std::string     modelPreprocessPath;
        std::string     modelPostprocessPath;
        std::string     modelDenoiserPath;

        Cuda::LightProbeGridParams grid;
    };
    
#ifdef _DEBUG     
    class FCNNProbeDenoiser : public ONNXModel
    {
    public:
        FCNNProbeDenoiser() = default;

        void Reinitialise() {}
        void Initialise(const FCNNProbeDenoiserParams&) {}
        void Evaluate(const FCNNProbeDenoiserParams&, const std::vector<Cuda::vec3>&, std::vector<Cuda::vec3>&) {}
    };

#else
    class FCNNProbeDenoiser : public ONNXModel
    {
    private:
        enum InitFlags : uint 
        { 
            kInitORT = 1,
            kInitTensors = 2,
            kInitTransforms = 4,
            kIsModelReady = kInitORT | kInitTensors | kInitTransforms
        };

    public:
        FCNNProbeDenoiser();
        ~FCNNProbeDenoiser();

        void Reinitialise();
        void Initialise(const FCNNProbeDenoiserParams& params);
        void Evaluate(const FCNNProbeDenoiserParams& params, const std::vector<Cuda::vec3>& inputData, std::vector<Cuda::vec3>& outputData);

    private:
        void Destroy();

        Tensor<float>                   m_unprocSHTensor;
        Tensor<float>                   m_procNoisySHTensor;
        Tensor<float>                   m_procDenoisedSHTensor;
        Tensor<float>                   m_unprocMaskTensor;
        Tensor<float>                   m_procMaskTensor;
        Tensor<int64_t>                 m_padTensor;
        Tensor<float>                   m_statTensor;

        OrtSession                      m_preProcSession;
        OrtSession                      m_denoiseSession;
        OrtSession                      m_postProcSession;

        std::vector<Cuda::vec3>         m_packedCoeffData;
        Cuda::LightProbeDataTransform   m_dataTransform;

        std::unique_ptr<Ort::Env>       m_ortEnvironment;
        std::unique_ptr<Ort::Session>   m_ortPreprocSession;
        std::unique_ptr<Ort::Session>   m_ortDenoiserSession;
        std::unique_ptr<Ort::Session>   m_ortPostprocSession;

        FCNNProbeDenoiserParams         m_params;
        uint                            m_initFlags;
    };

#endif

}
