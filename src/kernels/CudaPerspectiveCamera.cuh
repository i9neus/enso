#pragma once

#include "math/CudaMath.cuh"
#include "CudaRay.cuh"
#include "CudaCtx.cuh"

namespace Cuda
{
	namespace Device
	{
		class PerspectiveCamera
		{
		public:
			__device__ PerspectiveCamera();
			__device__ void CreateRay(CompressedRay& newRay, RenderCtx& renderCtx) const;

		private:
			bool		m_useHaltonSpectralSampler;
			vec2		m_cameraPos;
			vec2        m_cameraLook;
			vec2		m_cameraFLength;
			vec2        m_cameraFStop;
		};
	}
}