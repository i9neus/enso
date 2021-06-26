#include "CudaWavefrontTracer.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"
#include "generic/Assert.h"
#include "CudaAsset.cuh"
#include "CudaRay.cuh" 

#include "bxdfs/CudaLambert.cuh"
#include "tracables/CudaSphere.cuh"
#include "tracables/CudaPlane.cuh"
//#include "tracables/CudaCornell.cuh"
#include "tracables/CudaKIFS.cuh"
#include "materials/CudaMaterial.cuh"
#include "lights/CudaQuadLight.cuh"

#include "CudaPerspectiveCamera.cuh"
#include "CudaManagedArray.cuh"

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"
#include "CudaCtx.cuh"

#include "generic/JsonUtils.h"

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
		//BidirectionalTransform transform;
		//m_objects.cu_cornell->SetTransform(transform);
		//m_objects.cu_sphere->SetTransform(transform);
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
	}

	__device__ vec3 Device::WavefrontTracer::Shade(const Ray& incidentRay, const HitCtx& hitCtx, RenderCtx& renderCtx) const
	{
		if (renderCtx.depth >= 1) { return kZero; }

		vec3 albedo, incandescence;
		//m_objects.cu_simpleMaterial->Evaluate(hitCtx, albedo, incandescence);
		vec2 xi = renderCtx.Rand<2, 3>();

		// Sample the BRDF
		//if(xi.x < 0.75f)
		{			
			vec3 brdfDir;
			float brdfPdf;
			//if (!m_objects.cu_lambert->Sample(incidentRay, hitCtx, renderCtx, brdfDir, brdfPdf)) { return incandescence; }

			// Light evaluation
			/*if (xi.y < 0.5f)
			{

			}
			// Global illumination evaluation
			else
			{

			}*/

			const vec3 weight = incidentRay.weight * albedo;

			renderCtx.EmplaceRay(RayBasic(hitCtx.ExtantOrigin(), brdfDir), weight, brdfPdf, incidentRay.lambda, 0, renderCtx.depth);
		}
		/*else
		{
			vec3 brdfDir;
			float brdfPdf;
			if (m_objects.cu_lambert->Sample(incidentRay, hitCtx, renderCtx, brdfDir, brdfPdf))
			{
				const vec3 weight = incidentRay.weight * albedo;

				renderCtx.EmplaceRay(RayBasic(hitCtx.ExtantOrigin(), brdfDir), weight, brdfPdf, incidentRay.lambda, 0, renderCtx.depth);
			}
		}*/

		return incandescence;
	}

	__device__ void Device::WavefrontTracer::PreBlock() const
	{
		for (int i = 0; i < m_objects.cu_deviceTracables->Size(); i++)
		{
			(*m_objects.cu_deviceTracables)[i]->InitialiseKernelConstantData();
		}
	}

	__device__ void Device::WavefrontTracer::Trace(const uint rayIdx) const
	{		
		if (rayIdx >= m_objects.cu_deviceCompressedRayBuffer->Size()) { return; }

		CompressedRay& compressedRay = (*m_objects.cu_deviceCompressedRayBuffer)[rayIdx];
		Ray incidentRay(compressedRay);
		RenderCtx renderCtx(compressedRay.ViewportPos(), m_objects.viewportDims, m_wallTime, compressedRay.sampleIdx, compressedRay.depth);
		vec3 L(0.0f);
		const ivec2 viewportPos = compressedRay.ViewportPos(); // FIXME: Do an automatic cast				

		int depth = renderCtx.depth; 

		// INTERSECTION 
		HitCtx hitCtx;
		for (int i = 0; i < m_objects.cu_deviceTracables->Size(); i++)
		{
			(*m_objects.cu_deviceTracables)[i]->Intersect(incidentRay, hitCtx);
		}		

		// SHADE
		if (!hitCtx.isValid)
		{
			L += incidentRay.weight * vec3(1.0f);
		}
		else
		{
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

		//m_objects.cu_deviceAccumBuffer->At(viewportPos) = vec4(hitCtx.hit.p, -1.0f);
		//return;

		// FIXME: Do an automatic cast
		m_objects.cu_deviceAccumBuffer->Accumulate(viewportPos, L, renderCtx.depth, renderCtx.emplacedRay.IsAlive());
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

		/*for (auto& tracable : *m_hostTracables)	{ tracable.DestroyAsset(); }
		for (auto& light : *m_hostLights) { light.DestroyAsset(); }
		for (auto& material : *m_hostMaterials) { material.DestroyAsset(); }*/
		
		m_hostPerspectiveCamera.DestroyAsset();

		DestroyOnDevice(&cu_deviceData);
	}

	__host__ AssetHandle<Host::RenderObject> Host::WavefrontTracer::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
	{
		if (expectedType != AssetType::kIntegrator) { return AssetHandle<Host::RenderObject>(); }

		return AssetHandle<Host::RenderObject>(new Host::QuadLight(json), id);
	}

	__host__ Host::WavefrontTracer::WavefrontTracer(cudaStream_t hostStream) :
		cu_deviceData(nullptr),
		m_hostStream(hostStream),
		m_isDirty(true)
	{
		// Create the packed ray buffer
		/*m_hostCompressedRayBuffer = AssetHandle<Host::CompressedRayBuffer>("id_hostCompressedRayBuffer", 512 * 512, m_hostStream);

		// Create the accumulation buffer
		m_hostAccumBuffer = AssetHandle<Host::ImageRGBW>("id_hostAccumBuffer", 512, 512, m_hostStream);
		m_hostAccumBuffer->Clear(vec4(0.0f));

		m_hostTracables = AssetHandle<Host::AssetContainer<Host::Tracable>>("id_tracableContainer");
		m_hostMaterials = AssetHandle<Host::AssetContainer<Host::Material>>("id_materialContainer");
		m_hostLights = AssetHandle<Host::AssetContainer<Host::Light>>("id_lightContainer");
		m_hostBxDFs = AssetHandle<Host::AssetContainer<Host::BxDF>>("id_bxdfContainer");

		m_hostTracables->Push(AssetHandle<Host::Tracable>(new Host::Cornell(), "id_cornell"));
		m_hostTracables->Push(AssetHandle<Host::Tracable>(new Host::Sphere(), "id_sphere"));
		m_hostTracables->Push(AssetHandle<Host::Tracable>(new Host::Plane(BidirectionalTransform(vec3(0.0f), vec3(kHalfPi, 0.0f, 0.0f), vec3(1.0f)), false), "id_groundplane"));
		auto quadLightPlane = AssetHandle<Host::Tracable>(new Host::Plane(BidirectionalTransform(), true), "id_quadlightplane");
		m_hostTracables->Push(quadLightPlane);		
		m_hostTracables->Push(AssetHandle<Host::Tracable>(new Host::KIFS(), "id_kifs"));
		m_hostTracables->Synchronise();

		m_hostMaterials->Push(AssetHandle<Host::Material>(new Host::SimpleMaterial(), "id_simpleMaterial"));
		m_hostMaterials->Synchronise();

		m_hostLights->Push(AssetHandle<Host::Light>(new Host::QuadLight(quadLightPlane), "id_quadlight"));
		m_hostLights->Synchronise();

		m_hostPerspectiveCamera = AssetHandle<Host::PerspectiveCamera>(new Host::PerspectiveCamera(), "id_perspcamera");

		m_hostBxDFs->Push(AssetHandle<Host::BxDF>(new Host::LambertBRDF(), "id_lambert"));
		m_hostLights->Synchronise();*/

		checkCudaErrors(cudaDeviceSynchronize());

		// Create the wavefront tracer structure on the device
		/*m_hostObjects.cu_deviceAccumBuffer = m_hostAccumBuffer->GetDeviceInstance();
		m_hostObjects.cu_deviceCompressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
		//m_hostObjects.cu_pixelFlagsBuffer = m_hostPixelFlagsBuffer->GetDeviceInstance();
		m_hostObjects.cu_deviceTracables = m_hostTracables->GetDeviceInstance();
		m_hostObjects.viewportDims = m_hostAccumBuffer->GetHostInstance().Dimensions();*/

		cu_deviceData = InstantiateOnDeviceWithParams<Device::WavefrontTracer>(m_hostObjects);
		
		m_block = dim3(16, 16, 1);
		m_grid = dim3((m_hostAccumBuffer->GetHostInstance().Width() + 15) / 16, (m_hostAccumBuffer->GetHostInstance().Height() + 15) / 16, 1);
	}

	__host__ Host::WavefrontTracer::~WavefrontTracer() 
	{ 
		OnDestroyAsset(); 
	}

	__host__ void Host::WavefrontTracer::FromJson(const ::Json::Node& jsonNode)
	{
		/*m_hostTracables->OnJson(jsonNode);
		m_hostMaterials->OnJson(jsonNode);
		m_hostLights->OnJson(jsonNode);
		m_hostBxDFs->OnJson(jsonNode);

		m_hostPerspectiveCamera->OnJson(jsonNode);
		m_isDirty = true;*/
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
		if (kThreadIdx == 0)
		{
			tracer->PreBlock();
		}
		__syncthreads();

		tracer->Trace(kKernelIdx);
	}

	__global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::WavefrontTracer* tracer)
	{
		if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

		tracer->Composite(kKernelPos<ivec2>(), deviceOutputImage);
	}

	__host__ void Host::WavefrontTracer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage)
	{
		//std::printf("Composite! %i %i %i\n", m_grid.x, m_grid.y, m_grid.z);
	
		hostOutputImage->SignalSetWrite(m_hostStream);
		KernelComposite << < m_grid, m_block, 0, m_hostStream >> > (hostOutputImage->GetDeviceInstance(), cu_deviceData);
		hostOutputImage->SignalUnsetWrite(m_hostStream);
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