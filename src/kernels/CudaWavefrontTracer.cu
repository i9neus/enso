#include "CudaWavefrontTracer.cuh"
#include "generic/Assert.h"

namespace Cuda
{
	void WavefrontTracer::Initialise(Image* deviceImage)
	{
		cu_deviceImage = deviceImage;
	}

	void WavefrontTracer::Iterate()
	{
		Assert(cu_deviceImage != nullptr);
	}
}