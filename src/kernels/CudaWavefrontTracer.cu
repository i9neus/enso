#include "CudaWavefrontTracer.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"
#include "generic/Assert.h"
#include "CudaAsset.cuh"
#include "CudaRay.cuh" 

#include "bxdfs/CudaLambert.cuh"
#include "tracables/CudaSphere.cuh"
#include "tracables/CudaPlane.cuh"
#include "tracables/CudaCornell.cuh"
#include "tracables/CudaKIFS.cuh"
#include "materials/CudaMaterial.cuh"

#include "CudaPerspectiveCamera.cuh"
#include "CudaManagedArray.cuh"

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"
#include "CudaCtx.cuh"

namespace Cuda
{
	/*__device__ void Device::WavefrontTracer::OnSyncParams(const Device::WavefrontTracer::Params* params)
	{
		if (params) { m_objects = *params; }
	}*/

	__device__ Device::WavefrontTracer::WavefrontTracer(const Objects* cu_objects)
	{
		if (cu_objects) { m_objects = *cu_objects; }
	}

	__device__ void Device::WavefrontTracer::PreFrame(const float& wallTime, const int frameIdx)
	{
		m_wallTime = wallTime;
		m_frameIdx = frameIdx;

		//auto transform = CreateCompoundTransform(vec3(0.8f, 1.1f, 0.9f) * m_frameIdx / 100.0f);
		BidirectionalTransform transform;
		m_objects.cu_cornell->SetTransform(transform);
		m_objects.cu_sphere->SetTransform(transform);
	}

	__device__ void Device::WavefrontTracer::SeedRayBuffer(const ivec2& viewportPos) const
	{
		if (!IsValid(viewportPos)) { return; }

		CompressedRay& compressedRay = (*m_objects.cu_deviceCompressedRayBuffer)[viewportPos.y * 512 + viewportPos.x];

		if (!compressedRay.IsAlive())
		{
			RenderCtx renderCtx(viewportPos, m_objects.viewportDims, m_wallTime, compressedRay.sampleIdx + 1, 0);
			m_objects.cu_camera->CreateRay(compressedRay, renderCtx);
		}

		//cu_deviceAccumBuffer->At(viewportPos) = vec4(newRay.od.d, 1.0f);
	}

	__device__ vec3 Device::WavefrontTracer::Shade(const Ray& incidentRay, const HitCtx& hitCtx, RenderCtx& renderCtx) const
	{
		if (renderCtx.depth >= 1) { return kZero; }

		//const vec4 xi = renderCtx.Rand4();

		vec3 brdfDir;
		float brdfPdf;
		if (m_objects.cu_lambert->Sample(incidentRay, hitCtx, renderCtx, brdfDir, brdfPdf))
		{
			const vec3 weight = incidentRay.weight * 
								m_objects.cu_simpleMaterial->Evaluate(hitCtx);

			renderCtx.EmplaceRay(RayBasic(hitCtx.ExtantOrigin(), brdfDir), weight, brdfPdf, incidentRay.lambda, 0, renderCtx.depth);
		}

		return kZero;
	}

	__device__ void Device::WavefrontTracer::Trace(const uint rayIdx) const
	{
		if (rayIdx >= m_objects.cu_deviceCompressedRayBuffer->Size()) { return; }

		CompressedRay& compressedRay = (*m_objects.cu_deviceCompressedRayBuffer)[rayIdx];
		Ray incidentRay(compressedRay);
		RenderCtx renderCtx(compressedRay.ViewportPos(), m_objects.viewportDims, m_wallTime, compressedRay.sampleIdx, compressedRay.depth);
		vec3 L(0.0f);
		const vec2 viewportPos = vec2(compressedRay.ViewportPos()); // FIXME: Do an automatic cast

		int depth = renderCtx.depth; 

		// INTERSECTION 
		HitCtx hitCtx;
		//for (int i = 0; i < cu_deviceTracables->Size(); i++)
		{
			//m_objects.cu_cornell->Intersect(incidentRay, hitCtx);
			//m_objects.cu_sphere->Intersect(incidentRay, hitCtx);
			m_objects.cu_groundPlane->Intersect(incidentRay, hitCtx);
			m_objects.cu_kifs->Intersect(incidentRay, hitCtx);
		}

		// SHADE
		if (!hitCtx.isValid)
		{
			L += incidentRay.weight * vec3(1.0f);
		}
		else
		{
			//L += hitCtx.hit.n;// *0.5f + vec3(0.5f);
			L += Shade(incidentRay, hitCtx, renderCtx);
		}

		if (renderCtx.emplacedRay.IsAlive())
		{
			compressedRay = renderCtx.emplacedRay;
			//L += compressedRay.od.d * 0.5f + vec3(0.5f);
		}
		else
		{
			compressedRay.Kill();			
		}

		//L += incidentRay.od.d;
		//cu_deviceAccumBuffer->At(viewportPos) = 0.0f;

		// FIXME: Do an automatic cast
		m_objects.cu_deviceAccumBuffer->Accumulate(ivec2(viewportPos), L, renderCtx.depth, renderCtx.emplacedRay.IsAlive());
		//cu_deviceAccumBuffer->At(viewportPos) += vec4(L, 1.0f);
	}

	__device__ void Device::WavefrontTracer::Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const
	{
		if (viewportPos.x >= deviceOutputImage->Width() || viewportPos.y >= deviceOutputImage->Height() ||
			viewportPos.x >= m_objects.cu_deviceAccumBuffer->Width() || viewportPos.y >= m_objects.cu_deviceAccumBuffer->Height()) {
			return;
		}

		// If the texel weight is negative, the texel is ready to be rendered
		vec4& texel = m_objects.cu_deviceAccumBuffer->At(viewportPos);
		if (texel.w >= 0.0f) { return; }

		// Flip the weight back to positve
		texel.w = -texel.w;

		const vec3 rgb = texel.xyz / fmax(1.0f, texel.w);
		deviceOutputImage->At(viewportPos) = vec4(rgb, 1.0f);
	}

	__host__ void Host::WavefrontTracer::OnDestroyAsset()
	{
		if (!m_hostCompressedRayBuffer) { return; }
		
		m_hostCompressedRayBuffer.DestroyAsset();
		m_hostAccumBuffer.DestroyAsset();
		m_hostTracables.DestroyAsset();
		m_hostCornell.DestroyAsset();
		m_hostGroundPlane.DestroyAsset();
		m_hostSphere.DestroyAsset();
		m_hostKifs.DestroyAsset();
		m_hostSimpleMaterial.DestroyAsset();
		m_hostLambert.DestroyAsset();
		m_hostPerspectiveCamera.DestroyAsset();

		DestroyOnDevice(&cu_deviceData);
	}

	__host__ Host::WavefrontTracer::WavefrontTracer(cudaStream_t hostStream) :
		cu_deviceData(nullptr),
		m_hostStream(hostStream),
		m_isDirty(true)
	{
		// Create the packed ray buffer
		m_hostCompressedRayBuffer = AssetHandle<Host::CompressedRayBuffer>("id_hostCompressedRayBuffer", 512 * 512, m_hostStream);

		// Create the accumulation buffer
		m_hostAccumBuffer = AssetHandle<Host::ImageRGBW>("id_hostAccumBuffer", 512, 512, m_hostStream);
		m_hostAccumBuffer->Clear(vec4(0.0f));

		m_hostTracables = AssetHandle<Host::AssetContainer<Host::Tracable>>("id_tracableContainer");

		m_hostCornell = AssetHandle<Host::Cornell>(new Host::Cornell(), "id_cornell");
		m_hostSphere = AssetHandle<Host::Sphere>(new Host::Sphere(), "id_sphere");
		m_hostGroundPlane = AssetHandle<Host::Plane>(new Host::Plane(CreateCompoundTransform(vec3(kHalfPi, 0.0f, 0.0f)), false), "id_plane");
		m_hostSimpleMaterial = AssetHandle<Host::SimpleMaterial>(new Host::SimpleMaterial(), "id_simpleMaterial");
		m_hostKifs = AssetHandle<Host::KIFS>(new Host::KIFS(), "id_kifs");

		m_hostPerspectiveCamera = AssetHandle<Host::PerspectiveCamera>(new Host::PerspectiveCamera(), "id_perspcamera");

		//m_hostTracables->Push(newSphere);
		//m_hostTracables->Sync();

		m_hostLambert = AssetHandle<Host::LambertBRDF>(new Host::LambertBRDF(), "id_lambert");

		checkCudaErrors(cudaDeviceSynchronize());

		// Create the wavefront tracer structure on the device
		m_hostData.cu_deviceAccumBuffer = m_hostAccumBuffer->GetDeviceInstance();
		m_hostData.cu_deviceCompressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
		//m_hostData.cu_pixelFlagsBuffer = m_hostPixelFlagsBuffer->GetDeviceInstance();
		//cu_deviceTracables = m_hostTracables->GetDeviceInstance();
		m_hostData.viewportDims = m_hostAccumBuffer->GetHostInstance().Dimensions();
		m_hostData.cu_cornell = m_hostCornell->GetDeviceInstance();
		m_hostData.cu_sphere = m_hostSphere->GetDeviceInstance();
		m_hostData.cu_groundPlane = m_hostGroundPlane->GetDeviceInstance();
		m_hostData.cu_simpleMaterial = m_hostSimpleMaterial->GetDeviceInstance();
		m_hostData.cu_kifs = m_hostKifs->GetDeviceInstance();
		m_hostData.cu_camera = m_hostPerspectiveCamera->GetDeviceInstance();

		cu_deviceData = InstantiateOnDeviceWithParams<Device::WavefrontTracer>(m_hostData);
		
		m_block = dim3(16, 16, 1);
		m_grid = dim3((m_hostAccumBuffer->GetHostInstance().Width() + 15) / 16, (m_hostAccumBuffer->GetHostInstance().Height() + 15) / 16, 1);

		std::printf("%i, %i, %i\n", m_grid.x, m_grid.y, m_grid.z);
	}

	__host__ void Host::WavefrontTracer::OnJson(const Json::Node& jsonNode)
	{
		m_hostAccumBuffer->OnJson(jsonNode);
		m_hostTracables->OnJson(jsonNode);
		m_hostCornell->OnJson(jsonNode);
		m_hostSphere->OnJson(jsonNode);
		m_hostLambert->OnJson(jsonNode);
		m_hostGroundPlane->OnJson(jsonNode);
		m_hostSimpleMaterial->OnJson(jsonNode);
		m_hostKifs->OnJson(jsonNode);
		m_isDirty = true;
	}

	__global__ void KernelPreFrame(Device::WavefrontTracer* tracer, const float wallTime, const int frameIdx)
	{
		tracer->PreFrame(wallTime, frameIdx);
	}

	__global__ void KernelSeedRayBuffer(Device::WavefrontTracer* tracer)
	{
		tracer->SeedRayBuffer(kKernelPos<ivec2>());
	}

	__global__ void KernelTrace(Device::WavefrontTracer* tracer)
	{
		tracer->Trace(blockIdx.x * blockDim.x + threadIdx.x);
	}

	__global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::WavefrontTracer* tracer)
	{
		//if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

		tracer->Composite(kKernelPos<ivec2>(), deviceOutputImage);
	}

	__host__ void Host::WavefrontTracer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage)
	{
		//std::printf("Composite! %i %i %i\n", m_grid.x, m_grid.y, m_grid.z);
	
		KernelComposite << < m_grid, m_block, 0, m_hostStream >> > (hostOutputImage->GetDeviceInstance(), cu_deviceData);
	}

	__host__ void Host::WavefrontTracer::Iterate(const float wallTime, const float frameIdx)
	{
		//std::printf("Iterate! %f\n", wallTime);

		if (m_isDirty)
		{
			m_hostAccumBuffer->Clear(vec4(0.0f));
			m_hostCompressedRayBuffer->Clear(Cuda::CompressedRay());
			//m_hostPixelFlagsBuffer->Clear(0);
			m_isDirty = false;
		}
		
		KernelPreFrame << < 1, 1, 0, m_hostStream >> > (cu_deviceData, wallTime, frameIdx);

		KernelSeedRayBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData);

		KernelTrace << <  m_hostCompressedRayBuffer->NumBlocks(), m_hostCompressedRayBuffer->ThreadsPerBlock(), 0, m_hostStream >> > (cu_deviceData);
	}
}