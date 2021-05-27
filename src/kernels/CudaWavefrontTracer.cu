#include "CudaWavefrontTracer.cuh"
#include "generic/Assert.h"

namespace Cuda
{
	__global__ void KernelDrawTestPattern(Device::ImageRGBW* image)
	{
		unsigned int kx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int ky = blockIdx.y * blockDim.y + threadIdx.y;

		if (kx >= image->Width() || ky >= image->Height()) { return; }

		vec4 pixel;
		float shade = ((kx / 10) + (ky / 10)) % 2;		
		pixel.x = shade * float(kx) / float(image->Width());
		pixel.y = shade * float(ky) / float(image->Height());
		pixel.z = 0.5f;
		pixel.w = 1.0f;

		*(image->At(kx, ky)) += pixel;
	}

	__global__ void KernelSeedRayBuffer(Device::WavefrontTracer* tracer)
	{

	}

	__global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::WavefrontTracer* wavefrontTracer)
	{
		if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

		wavefrontTracer->Composite(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, deviceOutputImage);
	}

	void Device::WavefrontTracer::Composite(unsigned int kx, unsigned int ky, Device::ImageRGBA* deviceOutputImage) const
	{		
		if (kx >= deviceOutputImage->Width() || ky >= deviceOutputImage->Height() ||
			kx >= m_deviceAccumBuffer->Width() || ky >= m_deviceAccumBuffer->Height()) {
			return;
		}		

		vec4 pixel;
		float shade = ((kx / 10) + (ky / 10)) % 2;
		pixel.x = shade * float(kx) / float(deviceOutputImage->Width());
		pixel.y = shade * float(ky) / float(deviceOutputImage->Height());
		pixel.z = 0.5f;
		pixel.w = 1.0f;

		*(deviceOutputImage->At(kx, ky)) += pixel;
	}

	void Host::WavefrontTracer::OnDestroyAsset()
	{
		if (!m_hostPackedRayBuffer) { return; }
		
		m_hostPackedRayBuffer.DestroyAsset();
		m_hostAccumBuffer.DestroyAsset();
	}

	Host::WavefrontTracer::WavefrontTracer(cudaStream_t hostStream) : 
		Device::WavefrontTracer(),
		cu_deviceTracer(nullptr),
		m_hostStream(hostStream)
	{
		// Create the packed ray buffer
		m_hostPackedRayBuffer = Asset<Host::PackedRayBuffer>("id_hostPackedRayBuffer", 512, 512, m_hostStream);
		m_hostPackedRayBuffer->Clear(PackedRay()); 

		// Create the accumulation buffer
		m_hostAccumBuffer = Asset<Host::ImageRGBW>("id_hostAccumBuffer", 512, 512, m_hostStream);
		m_hostAccumBuffer->Clear(vec4(0.0f));

		// Create the wavefront tracer structure on the device
		m_deviceAccumBuffer = m_hostAccumBuffer->GetDeviceImage();
		m_devicePackedRayBuffer = m_hostPackedRayBuffer->GetDeviceImage();
		checkCudaErrors(cudaMalloc((void**)&cu_deviceTracer, sizeof(Device::WavefrontTracer)));
		checkCudaErrors(cudaMemcpy(cu_deviceTracer, this, sizeof(Device::WavefrontTracer), cudaMemcpyHostToDevice));
		
		m_block = dim3(16, 16, 1);
		m_grid = dim3((m_hostAccumBuffer->Width() + 15) / 16, (m_hostAccumBuffer->Height() + 15) / 16, 1);
	}

	void Host::WavefrontTracer::Composite(Asset<Host::ImageRGBA>& hostOutputImage)
	{
		KernelComposite << < m_grid, m_block, 0, m_hostStream >> > (hostOutputImage->GetDeviceImage(), cu_deviceTracer);
	}

	void Host::WavefrontTracer::Iterate()
	{
		std::printf("Iterate!\n");
		//KernelDrawTestPattern << < m_grid, m_block, 0, m_hostImage->GetHostStream() >> > (m_hostAccumBuffer->GetDeviceImage());
	}
}