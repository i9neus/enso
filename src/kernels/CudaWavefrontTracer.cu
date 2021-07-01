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
	__host__ void WavefrontTracerParams::ToJson(::Json::Node& node) const
	{

	}

	__host__ void WavefrontTracerParams::FromJson(const ::Json::Node& node, const uint flags)
	{

	}
	
	__device__ void Device::WavefrontTracer::Synchronise(const Device::WavefrontTracer::Objects& objects)
	{
		m_objects = objects;
	}

	__device__ void Device::WavefrontTracer::Synchronise(const WavefrontTracerParams& params)
	{
		m_params = params;
	}

	__device__ void Device::WavefrontTracer::PreFrame(const float& wallTime, const int frameIdx)
	{
		m_wallTime = wallTime;
		m_frameIdx = frameIdx;
	}

	__device__ void Device::WavefrontTracer::SeedRayBuffer(const ivec2& viewportPos) const
	{
		if (!IsValid(viewportPos)) { return; }

		CompressedRay& compressedRay = (*m_objects.cu_deviceCompressedRayBuffer)[viewportPos.y * 512 + viewportPos.x];

		if (!compressedRay.IsAlive())
		{
			compressedRay.viewport.x = viewportPos.x;
			compressedRay.viewport.y = viewportPos.y;
			compressedRay.sampleIdx = compressedRay.sampleIdx + 1;

			RenderCtx renderCtx(compressedRay, m_objects.viewportDims);
			m_objects.cu_camera->CreateRay(renderCtx);
		}
	}

	__device__ vec3 Device::WavefrontTracer::Shade(const Ray& incidentRay, const Device::Material& material, const HitCtx& hitCtx, RenderCtx& renderCtx) const
	{
		if (renderCtx.depth >= 1) { return kZero; }

		vec3 albedo, incandescence;
		material.Evaluate(hitCtx, albedo, incandescence);
		vec2 xi = renderCtx.Rand<2, 3>();

		const BxDF* bxdf = material.GetBoundBxDF();
		if (!bxdf) { return kPink; }

		// Sample the BRDF
		//if(xi.x < 0.75f)
		{			
			vec3 brdfDir;
			float brdfPdf;
			if (!bxdf->Sample(incidentRay, hitCtx, renderCtx, brdfDir, brdfPdf)) { return incandescence; }

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
		RenderCtx renderCtx(compressedRay, m_objects.viewportDims);

		compressedRay.Reset();
		
		//m_objects.cu_deviceAccumBuffer->At(renderCtx.viewportPos) = vec4(renderCtx.viewportPos.x / 512.0f, renderCtx.viewportPos.y / 512.0f, 0.0f, -1.0f);
		//return;

		// INTERSECTION 
		HitCtx hitCtx;
		auto& tracables = *m_objects.cu_deviceTracables;
		Device::Tracable* hitObject = nullptr;
		for (int i = 0; i < tracables.Size(); i++)
		{
			if (tracables[i]->Intersect(incidentRay, hitCtx))
			{
				hitObject = tracables[i];
			}
		}		

		// SHADE
		vec3 L(0.0f);
		if (!hitObject)
		{
			L += incidentRay.weight * vec3(1.0f);
		}
		else
		{
			const Device::Material* hitMaterial = hitObject->GetBoundMaterial();			
			if (!hitMaterial)
			{
				// If no material is bound to this tracable, shade pink to get people's attention
				L += kPink;
			}
			else
			{
				L += Shade(incidentRay, *hitMaterial, hitCtx, renderCtx);
			}
		}

		if (renderCtx.emplacedRay.IsAlive())
		{
			compressedRay = renderCtx.emplacedRay;
		}

		// FIXME: Do an automatic cast
		m_objects.cu_deviceAccumBuffer->Accumulate(renderCtx.viewportPos, L, renderCtx.depth, renderCtx.emplacedRay.IsAlive());
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
		m_hostLights.DestroyAsset();

		DestroyOnDevice(cu_deviceData);
	}

	__host__ AssetHandle<Host::RenderObject> Host::WavefrontTracer::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
	{
		if (expectedType != AssetType::kIntegrator) { return AssetHandle<Host::RenderObject>(); }

		return AssetHandle<Host::RenderObject>(new Host::WavefrontTracer(json), id);
	}

	__host__ Host::WavefrontTracer::WavefrontTracer(const ::Json::Node& node) :
		cu_deviceData(nullptr),
		m_isDirty(true)
	{
		// Create the packed ray buffer
		m_hostCompressedRayBuffer = AssetHandle<Host::CompressedRayBuffer>("id_hostCompressedRayBuffer", 512 * 512, m_hostStream);
		m_hostCompressedRayBuffer->Clear(CompressedRay());

		// Create the accumulation buffer
		m_hostAccumBuffer = AssetHandle<Host::ImageRGBW>("id_hostAccumBuffer", 512, 512, m_hostStream);
		m_hostAccumBuffer->Clear(vec4(0.0f));

		m_hostTracables = AssetHandle<Host::AssetContainer<Host::Tracable>>("wavefront_tracablesContainer");
		m_hostLights = AssetHandle<Host::AssetContainer<Host::Light>>("wavefront_lightsContainer");	

		cu_deviceData = InstantiateOnDevice<Device::WavefrontTracer>();
		FromJson(node, ::Json::kRequiredWarn);
		
		m_block = dim3(16, 16, 1);
		m_grid = dim3((m_hostAccumBuffer->GetHostInstance().Width() + 15) / 16, (m_hostAccumBuffer->GetHostInstance().Height() + 15) / 16, 1);
	}

	__host__ Host::WavefrontTracer::~WavefrontTracer() 
	{ 
		OnDestroyAsset(); 
	}

	__host__ void Host::WavefrontTracer::Bind(RenderObjectContainer& sceneObjects)
	{
		Log::Indent indent;
		for (auto& object : sceneObjects)
		{
			switch (object->GetAssetType())
			{
			case AssetType::kTracable:
				Log::Debug("Linked tracable '%s' to wavefront tracer.\n", object->GetAssetID());
				m_hostTracables->Push(object.DynamicCast<Tracable>()); break;			
			case AssetType::kLight:
				Log::Debug("Linked light '%s' to wavefront tracer.\n", object->GetAssetID());
				m_hostLights->Push(object.DynamicCast<Light>()); break;
			}
		}		

		// Synchronise the container objects managed by this instance
		m_hostTracables->Synchronise();
		m_hostLights->Synchronise();

		// Synchronise the wavefront tracer structure on the device
		m_hostObjects.cu_deviceAccumBuffer = m_hostAccumBuffer->GetDeviceInstance();
		m_hostObjects.cu_deviceCompressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
		m_hostObjects.cu_deviceTracables = m_hostTracables->GetDeviceInstance();
		m_hostObjects.cu_deviceLights = m_hostLights->GetDeviceInstance();
		m_hostObjects.viewportDims = m_hostAccumBuffer->GetHostInstance().Dimensions();

		m_cameraAsset = GetAssetHandleForBinding<Host::WavefrontTracer, Host::PerspectiveCamera>(sceneObjects, m_cameraId);
		if (m_cameraAsset)
		{
			m_hostObjects.cu_camera = m_cameraAsset->GetDeviceInstance();
		}

		SynchroniseObjects(cu_deviceData, m_hostObjects);
		Log::Write("Bound tracables and lights to wavefront tracer '%s'.\n", GetAssetID());
	}

	__host__ void Host::WavefrontTracer::FromJson(const ::Json::Node& parentNode, const uint flags)
	{		
		Host::RenderObject::FromJson(parentNode, flags);

		SynchroniseObjects(cu_deviceData, WavefrontTracerParams(parentNode, flags));

		parentNode.GetValue("camera", m_cameraId, flags);
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
		if (kThreadIdx == 0)
		{
			tracer->PreBlock();
		}
		__syncthreads();

		tracer->Trace(kKernelIdx);
	}

	__global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::WavefrontTracer* tracer)
	{
		//if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

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