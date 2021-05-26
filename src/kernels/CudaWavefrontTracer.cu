#include "CudaWavefrontTracer.cuh"
#include "generic/Assert.h"

namespace Cuda
{
	__global__ void KernelDrawTestPattern(DeviceImage* image)
	{
		if (*(image->AccessSignal()) != DeviceImage::kWriteLocked) { return; }

		unsigned int kx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int ky = blockIdx.y * blockDim.y + threadIdx.y;

		if (kx >= image->Width() || ky >= image->Height()) { return; }

		float4 pixel;
		float shade = ((kx / 10) + (ky / 10)) % 2;
		pixel.x = shade * float(kx) / float(image->Width());
		pixel.y = shade * float(ky) / float(image->Height());
		pixel.z = 0.0f;
		pixel.w = 1.0f;

		*(image->At(kx, ky)) = pixel;
	}
	
	void HostWavefrontTracer::Initialise(HostImage* hostImage)
	{
		m_hostImage = hostImage;
	}

	void HostWavefrontTracer::Iterate()
	{
		Assert(m_hostImage != nullptr);

		std::printf("Iterate!\n");

		dim3 block(16, 16, 1);
		dim3 grid(m_hostImage->Width() / 16, m_hostImage->Height() / 16, 1);

		m_hostImage->SignalSetWrite(m_hostImage->GetHostStream());
		KernelDrawTestPattern << < grid, block, 0, m_hostImage->GetHostStream() >> > (m_hostImage->GetDeviceImage());
		m_hostImage->SignalUnsetWrite(m_hostImage->GetHostStream());
	}
}