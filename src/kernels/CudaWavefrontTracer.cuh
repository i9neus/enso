#pragma once

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"

namespace Cuda
{
	class WavefrontTracer
	{
	private:
		Image*		cu_deviceImage;

	public:		
		WavefrontTracer() = default;

		void Initialise(Image* deviceImage);

		void Iterate();

	};
}