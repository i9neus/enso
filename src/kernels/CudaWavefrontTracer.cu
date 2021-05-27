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
		tracer->SeedRayBuffer(ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y));
	}

	__global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::WavefrontTracer* tracer)
	{
		if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

		tracer->Composite(ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y), deviceOutputImage);
	}

	void Device::WavefrontTracer::SeedRayBuffer(ivec2 fragCoord)
	{

	}

	void Device::WavefrontTracer::Composite(const ivec2 fragCoord, Device::ImageRGBA* deviceOutputImage) const
	{		
		if (fragCoord.x >= deviceOutputImage->Width() || fragCoord.y >= deviceOutputImage->Height() ||
			fragCoord.x >= m_deviceAccumBuffer->Width() || fragCoord.y >= m_deviceAccumBuffer->Height()) {
			return;
		}		

		vec4 pixel;
		float shade = ((fragCoord.x / 10) + (fragCoord.y / 10)) % 2;
		pixel.x = shade * float(fragCoord.x) / float(deviceOutputImage->Width());
		pixel.y = shade * float(fragCoord.y) / float(deviceOutputImage->Height());
		pixel.z = 0.5f;
		pixel.w = 1.0f;

		*(deviceOutputImage->At(fragCoord.x, fragCoord.y)) += pixel;
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
		KernelSeedRayBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceTracer);
	}
}