#pragma once

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"

namespace Cuda
{
	class HostWavefrontTracer
	{
	private:
		HostImage* m_hostImage;

	public:
		HostWavefrontTracer() = default;

		void Initialise(HostImage* hostImage);

		void Iterate();
	};
}