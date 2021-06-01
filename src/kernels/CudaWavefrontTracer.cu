#include "CudaWavefrontTracer.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"
#include "generic/Assert.h"
#include "CudaAsset.cuh"

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
		cu_deviceAccumBuffer = nullptr;
		cu_deviceCompressedRayBuffer = nullptr;		 
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
		if (!IsValid(viewportPos)) { return; }
		
		CompressedRay& packedRay = cu_deviceCompressedRayBuffer->At(viewportPos);

		if (!packedRay.IsAlive())
		{
			RenderCtx renderCtx = CreateRenderCtx(viewportPos, 0u);
			m_camera.CreateRay(packedRay, renderCtx);
			packedRay.SetAlive();
		}

		//cu_deviceAccumBuffer->At(viewportPos) = vec4(newRay.od.d, 1.0f);
	}

	__device__ void Device::WavefrontTracer::Trace(const ivec2& viewportPos) const
	{
		if (!IsValid(viewportPos)) { return; }
		
		Ray ray = DeriveRay(cu_deviceCompressedRayBuffer->At(viewportPos));
		HitCtx hit;

		//for (int i = 0; i < cu_deviceTracables->Size(); i++)
		{
			if(cu_sphere->Intersect(ray, hit))
			{
				cu_deviceAccumBuffer->Accumulate(viewportPos, hit.n);
			}
		}
	}

	__device__ void Device::WavefrontTracer::Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const
	{		
		if (viewportPos.x >= deviceOutputImage->Width() || viewportPos.y >= deviceOutputImage->Height() ||
			viewportPos.x >= cu_deviceAccumBuffer->Width() || viewportPos.y >= cu_deviceAccumBuffer->Height()) {
			return;
		}

		vec4 texel = cu_deviceAccumBuffer->At(viewportPos);
		texel.xyz /= fmax(1.0f, texel.w);
		texel.w = 1.0f;

		deviceOutputImage->At(viewportPos) = texel;
	}

	__host__ void Host::WavefrontTracer::OnDestroyAsset()
	{
		if (!m_hostCompressedRayBuffer) { return; }
		
		m_hostCompressedRayBuffer.DestroyAsset();
		m_hostAccumBuffer.DestroyAsset();
		m_hostTracables.DestroyAsset();
		m_hostSphere.DestroyAsset();

		DestroyOnDevice(&cu_deviceData);
	}

	__host__ Host::WavefrontTracer::WavefrontTracer(cudaStream_t hostStream) :
		cu_deviceData(nullptr),
		m_hostStream(hostStream)
	{
		// Create the packed ray buffer
		m_hostCompressedRayBuffer = AssetHandle<Host::CompressedRayBuffer>("id_hostCompressedRayBuffer", 512, 512, m_hostStream);

		// Create the accumulation buffer
		m_hostAccumBuffer = AssetHandle<Host::ImageRGBW>("id_hostAccumBuffer", 512, 512, m_hostStream);
		m_hostAccumBuffer->Clear(vec4(0.0f));

		m_hostTracables = AssetHandle<Host::AssetContainer<Host::Tracable>>("id_tracableContainer");

		m_hostSphere = AssetHandle<Host::Sphere>(new Host::Sphere(vec3(0.0f), 1.0f), "id_sphere");
		//m_hostTracables->Push(newSphere);		
		//m_hostTracables->Sync();

		checkCudaErrors(cudaDeviceSynchronize());

		// Create the wavefront tracer structure on the device
		m_hostData.cu_deviceAccumBuffer = m_hostAccumBuffer->GetDeviceInstance();
		m_hostData.cu_deviceCompressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
		//cu_deviceTracables = m_hostTracables->GetDeviceInstance();
		m_hostData.m_viewportDims = m_hostAccumBuffer->GetHostInstance().Dimensions();
		m_hostData.cu_sphere = m_hostSphere->GetDeviceInstance();

		InstantiateOnDevice(&cu_deviceData, m_hostData.cu_deviceAccumBuffer,
								 			m_hostData.cu_deviceCompressedRayBuffer, 
											m_hostData.cu_sphere,
											m_hostData.m_viewportDims);
		
		m_block = dim3(16, 16, 1);
		m_grid = dim3((m_hostAccumBuffer->GetHostInstance().Width() + 15) / 16, (m_hostAccumBuffer->GetHostInstance().Height() + 15) / 16, 1);

		std::printf("%i, %i, %i\n", m_grid.x, m_grid.y, m_grid.z);
	}

	__host__ void Host::WavefrontTracer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage)
	{
		std::printf("Composite! %i %i %i\n", m_grid.x, m_grid.y, m_grid.z);
	
		KernelComposite << < m_grid, m_block, 0, m_hostStream >> > (hostOutputImage->GetDeviceInstance(), cu_deviceData);
	}

	__host__ void Host::WavefrontTracer::Iterate()
	{
		std::printf("Iterate!\n");

		KernelSeedRayBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData);

		KernelTrace << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData);
	}
}