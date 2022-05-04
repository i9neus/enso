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
}

#endif 