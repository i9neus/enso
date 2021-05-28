#include "CudaWavefrontTracer.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"
#include "generic/Assert.h"

namespace Cuda
{
	__global__ void KernelSeedRayBuffer(Device::WavefrontTracer* tracer)
	{
		tracer->SeedRayBuffer(KERNEL_COORDS_IVEC2);
	}

	__global__ void KernelTrace(Device::WavefrontTracer* tracer)
	{
		tracer->Trace(KERNEL_COORDS_IVEC2);
	}

	__global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::WavefrontTracer* tracer)
	{
		//if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

		tracer->Composite(KERNEL_COORDS_IVEC2, deviceOutputImage);
	}

	Device::WavefrontTracer::WavefrontTracer()
	{
		m_deviceAccumBuffer = nullptr;
		m_deviceCompressedRayBuffer = nullptr;		 
	}

	__device__ Device::RenderCtx Device::WavefrontTracer::CreateRenderCtx(const ivec2& viewportPos, const uint depth) const
	{
		int seed = int(hashOf(depth, uint(viewportPos.x), uint(viewportPos.y)));
		
		RenderCtx ctx;
		ctx.pcg.Initialise(seed);
		ctx.viewportPos = viewportPos;
		ctx.viewportDims = m_viewportDims;

		return ctx;
	}

	__device__ void Device::WavefrontTracer::SeedRayBuffer(const ivec2& viewportPos) const
	{
		CompressedRay& packedRay = m_deviceCompressedRayBuffer->At(viewportPos);

		if (!packedRay.IsAlive())
		{
			RenderCtx renderCtx = CreateRenderCtx(viewportPos, 0u);
			m_camera.CreateRay(packedRay, renderCtx);
			packedRay.SetAlive();
		}

		//m_deviceAccumBuffer->At(viewportPos) = vec4(newRay.od.d, 1.0f);
	}

	__device__ void Device::WavefrontTracer::Trace(const ivec2& viewportPos) const
	{
		CompressedRay& packedRay = m_deviceCompressedRayBuffer->At(viewportPos);
		Ray ray(packedRay);


	}

	__device__ void Device::WavefrontTracer::Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const
	{		
		if (viewportPos.x >= deviceOutputImage->Width() || viewportPos.y >= deviceOutputImage->Height() ||
			viewportPos.x >= m_deviceAccumBuffer->Width() || viewportPos.y >= m_deviceAccumBuffer->Height()) {
			return;
		}

		vec4 texel = m_deviceAccumBuffer->At(viewportPos);
		texel.xyz /= fmax(1.0f, texel.w);
		texel.w = 1.0f;

		deviceOutputImage->At(viewportPos) = texel;
	}

	__host__ void Host::WavefrontTracer::OnDestroyAsset()
	{
		if (!m_hostCompressedRayBuffer) { return; }
		
		m_hostCompressedRayBuffer.DestroyAsset();
		m_hostAccumBuffer.DestroyAsset();

		SafeFreeDeviceMemory(&cu_deviceTracer);
	}

	__host__ Host::WavefrontTracer::WavefrontTracer(cudaStream_t hostStream) :
		Device::WavefrontTracer(),
		cu_deviceTracer(nullptr),
		m_hostStream(hostStream)
	{
		// Create the packed ray buffer
		m_hostCompressedRayBuffer = Asset<Host::CompressedRayBuffer>("id_hostCompressedRayBuffer", 512, 512, m_hostStream);

		// Create the accumulation buffer
		m_hostAccumBuffer = Asset<Host::ImageRGBW>("id_hostAccumBuffer", 512, 512, m_hostStream);

		//m_hostTracables = Asset<Host::Array<Device::Tracable*>>("id_hostAccumBuffer");

		checkCudaErrors(cudaDeviceSynchronize());

		// Create the wavefront tracer structure on the device
		m_deviceAccumBuffer = m_hostAccumBuffer->GetDeviceInstance();
		m_deviceCompressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
		m_viewportDims = m_hostAccumBuffer->Dimensions();

		checkCudaErrors(cudaMalloc((void**)&cu_deviceTracer, sizeof(Device::WavefrontTracer)));
		checkCudaErrors(cudaMemcpy(cu_deviceTracer, static_cast<Device::WavefrontTracer*>(this), sizeof(Device::WavefrontTracer), cudaMemcpyHostToDevice));
		
		m_block = dim3(16, 16, 1);
		m_grid = dim3((m_hostAccumBuffer->Width() + 15) / 16, (m_hostAccumBuffer->Height() + 15) / 16, 1);
	}

	__host__ void Host::WavefrontTracer::Composite(Asset<Host::ImageRGBA>& hostOutputImage)
	{
		std::printf("Composite! %i %i %i\n", m_grid.x, m_grid.y, m_grid.z);
	
		KernelComposite << < m_grid, m_block, 0, m_hostStream >> > (hostOutputImage->GetDeviceInstance(), cu_deviceTracer);
	}

	__host__ void Host::WavefrontTracer::Iterate()
	{
		std::printf("Iterate!\n");

		KernelSeedRayBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceTracer);
	}
}