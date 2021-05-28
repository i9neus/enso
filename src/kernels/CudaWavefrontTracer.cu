#include "CudaWavefrontTracer.cuh"
#include "CudaCtx.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"
#include "generic/Assert.h"

namespace Cuda
{
	__global__ void KernelSeedRayBuffer(Device::WavefrontTracer* tracer)
	{
		tracer->SeedRayBuffer(KERNEL_COORDS_IVEC2);
	}

	__global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::WavefrontTracer* tracer)
	{
		//if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

		tracer->Composite(KERNEL_COORDS_IVEC2, deviceOutputImage);
	}

	 Device::WavefrontTracer::WavefrontTracer()
	{
		m_deviceAccumBuffer = nullptr;
		m_devicePackedRayBuffer = nullptr;
		 
	}

	__device__ void Device::WavefrontTracer::SeedRayBuffer(ivec2 viewportPos)
	{
		int seed = int(hashOf(uint(viewportPos.x), uint(viewportPos.y)));
		
		RenderCtx renderCtx;
		renderCtx.pcg.Initialise(seed);
		renderCtx.viewportPos = viewportPos;
		renderCtx.viewportDims = m_viewportDims;

		PackedRay newRay = m_camera.CreateRay(renderCtx);

		m_deviceAccumBuffer->At(viewportPos) = vec4(newRay.od.d, 1.0f);
	}

	__device__ void Device::WavefrontTracer::Composite(const ivec2 viewportPos, Device::ImageRGBA* deviceOutputImage) const
	{		
		if (viewportPos.x >= deviceOutputImage->Width() || viewportPos.y >= deviceOutputImage->Height() ||
			viewportPos.x >= m_deviceAccumBuffer->Width() || viewportPos.y >= m_deviceAccumBuffer->Height()) {
			return;
		}

		vec4 texel = m_deviceAccumBuffer->At(viewportPos);
		texel.xyz /= max(1.0f, texel.w);
		texel.w = 1.0f;

		deviceOutputImage->At(viewportPos) = texel;
	}

	__host__ void Host::WavefrontTracer::OnDestroyAsset()
	{
		if (!m_hostPackedRayBuffer) { return; }

		checkCudaErrors(cudaFree(cu_deviceTracer));
		
		m_hostPackedRayBuffer.DestroyAsset();
		m_hostAccumBuffer.DestroyAsset();
	}

	__host__ Host::WavefrontTracer::WavefrontTracer(cudaStream_t hostStream) :
		Device::WavefrontTracer(),
		cu_deviceTracer(nullptr),
		m_hostStream(hostStream)
	{
		// Create the packed ray buffer
		m_hostPackedRayBuffer = Asset<Host::PackedRayBuffer>("id_hostPackedRayBuffer", 512, 512, m_hostStream);

		// Create the accumulation buffer
		m_hostAccumBuffer = Asset<Host::ImageRGBW>("id_hostAccumBuffer", 512, 512, m_hostStream);

		checkCudaErrors(cudaDeviceSynchronize());

		// Create the wavefront tracer structure on the device
		m_deviceAccumBuffer = m_hostAccumBuffer->GetDeviceImage();
		m_devicePackedRayBuffer = m_hostPackedRayBuffer->GetDeviceImage();
		m_viewportDims = m_hostAccumBuffer->Dimensions();

		checkCudaErrors(cudaMalloc((void**)&cu_deviceTracer, sizeof(Device::WavefrontTracer)));
		checkCudaErrors(cudaMemcpy(cu_deviceTracer, static_cast<Device::WavefrontTracer*>(this), sizeof(Device::WavefrontTracer), cudaMemcpyHostToDevice));
		
		m_block = dim3(16, 16, 1);
		m_grid = dim3((m_hostAccumBuffer->Width() + 15) / 16, (m_hostAccumBuffer->Height() + 15) / 16, 1);
	}

	__host__ void Host::WavefrontTracer::Composite(Asset<Host::ImageRGBA>& hostOutputImage)
	{
		std::printf("Composite! %i %i %i\n", m_grid.x, m_grid.y, m_grid.z);
	
		KernelComposite << < m_grid, m_block, 0, m_hostStream >> > (hostOutputImage->GetDeviceImage(), cu_deviceTracer);
	}

	__host__ void Host::WavefrontTracer::Iterate()
	{
		std::printf("Iterate!\n");

		KernelSeedRayBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceTracer);
	}
}