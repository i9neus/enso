#pragma once

#include "generic/StdIncludes.h"
#include "generic/D3DIncludes.h"
#include "generic/Math.h"
#include <cuda_runtime.h>
#include "generic/JsonUtils.h"

#include "kernels/CudaImage.cuh"
#include "kernels/CudaWavefrontTracer.cuh"
#include "kernels/CudaAsset.cuh"
#include "kernels/CudaRenderObject.cuh"

/*class SceneAssetContainer
{
	
private:
	Cuda::AssetHandle<Cuda::RenderObjectContainer>		m_renderObjects;	

	Json::Document										m_renderParamsJson;
	Json::Document										m_sceneJson;
};*/

using SceneDescription = Cuda::AssetHandle<Cuda::RenderObjectContainer>;
using ScenePeripheralContainer = Cuda::AssetHandle<Cuda::RenderObjectContainer>;

class SceneGenerator
{
public:
	SceneGenerator();

	void Initialise(Json::Document& configJson);

	void Permute(SceneDescription& outputScene);

private:
	Json::Document				m_sceneTemplate;
};
