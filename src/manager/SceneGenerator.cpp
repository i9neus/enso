#include "SceneGenerator.h"

#include "dx12/SecurityAttributes.h"
#include "dx12/DXSampleHelper.h"
#include "thirdparty/nvidia/helper_cuda.h"

#include "kernels/CudaTests.cuh"
#include "kernels/CudaCommonIncludes.cuh"
#include "kernels/CudaRenderObjectFactory.cuh"
#include "kernels/CudaWavefrontTracer.cuh"
#include "kernels/CudaAssetContainer.cuh"
#include "kernels/tracables/CudaTracable.cuh"
#include "kernels/cameras/CudaCamera.cuh"

#include "io/USDIO.h"

SceneGenerator::SceneGenerator()
{
}

void SceneGenerator::Initialise(Json::Document& configJson)
{

}

void SceneGenerator::Permute(SceneDescription& outputScene)
{

}