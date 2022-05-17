#include "OnnxModel.h"

#ifndef _DEBUG

#include <onnxruntime/onnxruntime_cxx_api.h>

namespace ONNX
{
    template<typename Type>
    Tensor<Type>::~Tensor()
    {
        Destroy();
    }

    template<typename Type>
    Tensor<Type>::Tensor()
    {
        ort = std::make_unique<Ort::Value>(nullptr);
    }

    template<typename Type>
    void Tensor<Type>::Initialise(const std::vector<int64_t>& sh)
    {
        // Allocate host memory
        int numElements = 1;
        shape = sh;
        for (int idx = 0; idx < shape.size(); ++idx) 
        { 
            AssertMsgFmt(shape[idx] > 0, "Illegal tensor dimension %i at index %i.", shape[idx], idx);
            numElements *= shape[idx];
        }
        data.resize(numElements);

        // Allocate the ONNX tensor
        auto ortAllocatorInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        ort->release();
        *ort = Ort::Value::CreateTensor<Type>(ortAllocatorInfo, data.data(), data.size(), shape.data(), shape.size());
    }

    template<typename Type>
    void Tensor<Type>::Destroy()
    {
        ort->release();
    }
}

#endif