#pragma once

#include "generic/StdIncludes.h"
#include "kernels/math/CudaMath.cuh"

// Forward declare a bunch of structures used by ONNX runtime
namespace Ort
{
    struct Env;
    struct Value;
    struct Session;
}

namespace ONNX
{
    template<typename Type>
    struct Tensor
    {
        Tensor();
        ~Tensor();
        Tensor(Tensor&) = delete;
        Tensor(Tensor&& other) = delete;

        void Initialise(const std::vector<int64_t>& shape);
        void Destroy();

        std::unique_ptr<Ort::Value>     ort;
        std::vector<Type>               data;
        std::vector<int64_t>            shape;
    };

    template struct Tensor<float>;
    template struct Tensor<int64_t>;
    
    class ONNXModel
    {
    protected:
        ONNXModel() = default;
    };
}
