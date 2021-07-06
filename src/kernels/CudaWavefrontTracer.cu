#include "CudaWavefrontTracer.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"
#include "generic/Assert.h"
#include "CudaAsset.cuh"
#include "CudaRay.cuh" 

#include "bxdfs/CudaLambert.cuh"
#include "tracables/CudaSphere.cuh"
#include "tracables/CudaPlane.cuh"
#include "tracables/CudaCornellBox.cuh"
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
	__host__ WavefrontTracerParams::WavefrontTracerParams() : 
		maxDepth(1),
		ambientRadiance(0.0f),
		debugNormals(false),
		importanceMode(kImportanceMIS)
	{
	}
	
	__host__ void WavefrontTracerParams::ToJson(::Json::Node& node) const
	{
		node.AddValue("maxDepth", maxDepth);
		node.AddArray("ambientRadiance", std::vector<float>({ ambientRadiance.x, ambientRadiance.y, ambientRadiance.z }));
		node.AddValue("debugNormals", debugNormals);

		const std::vector<std::string> importanceModeIds({ "mis", "light", "bxdf" });
		node.AddEnumeratedParameter("importanceMode", importanceModeIds, importanceMode);
	}

	__host__ void WavefrontTracerParams::FromJson(const ::Json::Node& node, const uint flags)
	{
		node.GetValue("maxDepth", maxDepth, flags);
		node.GetVector("ambientRadiance", ambientRadiance, ::Json::kSilent);
		node.GetValue("debugNormals", debugNormals, flags);

		const std::vector<std::string> importanceModeIds({ "mis", "light", "bxdf" });
		node.GetEnumeratedParameter("importanceMode", importanceModeIds, importanceMode, flags);
	}

	__host__ bool WavefrontTracerParams::operator==(const WavefrontTracerParams& rhs) const
	{
		return maxDepth == rhs.maxDepth &&
			ambientRadiance == rhs.ambientRadiance;
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
			compressedRay.sampleIdx++;
			compressedRay.depth = 0;

			RenderCtx renderCtx(compressedRay, m_objects.viewportDims);
			m_objects.cu_camera->CreateRay(renderCtx);
		}	
	}
	
	__device__ __forceinline__ float PowerHeuristic(float pdf1, float pdf2)
	{
		return 2.0f * sqr(pdf1) / (sqr(pdf1) + sqr(pdf2));
	}

	__device__ uchar Device::WavefrontTracer::GetImportanceMode(const RenderCtx& ctx) const
	{
		return m_params.importanceMode;
		//return (ctx.emplacedRay.viewport.x < 256) ? m_params.importanceMode : kImportanceMIS;
	}

	__device__ vec3 Device::WavefrontTracer::Shade(const Ray& incidentRay, const Device::Material& material, const HitCtx& hitCtx, RenderCtx& renderCtx) const
	{	
		vec3 albedo, incandescence;
		material.Evaluate(hitCtx, albedo, incandescence);

		if (renderCtx.depth >= m_params.maxDepth) { return incandescence; }

		vec2 xi = renderCtx.Rand<2, 3>();

		const BxDF* bxdf = material.GetBoundBxDF();
		if (!bxdf) 
		{ 
			return incandescence; 
		}

		// If there are no lights in this scene, always sample the BxDF
		if (GetImportanceMode(renderCtx) == kImportanceBxDF || m_objects.cu_deviceLights->Size() == 0)
		{
			xi.x *= 0.5f;
		}

		// Indirect light sampling
		if(xi.x < 0.5f)
		{			
			vec3 extantDir;
			float pdfBxDF;
			if (bxdf->Sample(incidentRay, hitCtx, renderCtx, extantDir, pdfBxDF))
			{
				vec3 L = renderCtx.emplacedRay.weight * albedo;
				if (GetImportanceMode(renderCtx) != kImportanceBxDF) { L *= 2.0f; }

				renderCtx.EmplaceIndirectSample(RayBasic(hitCtx.ExtantOrigin(), extantDir), L);
			}
		}
		// Direct light sampling
		else
		{		
			// Rescale the random number
			xi.x = (GetImportanceMode(renderCtx) == kImportanceLight) ? 0.0f : (xi.x * 2.0f - 1.0f);
			
			// Randomly select a light
			const int lightIdx = min(m_objects.cu_deviceLights->Size() - 1, uint(xi.y * m_objects.cu_deviceLights->Size()));
			const Light& light = *(*m_objects.cu_deviceLights)[lightIdx];

			float pdfBxDF, pdfLight;
			vec3 extantDir, L;

			// Sample the light
			if (xi.x < 0.5f)
			{
				if (light.Sample(incidentRay, hitCtx, renderCtx, extantDir, L, pdfLight))
				{
					float weightBxDF;
					bxdf->Evaluate(incidentRay.od.d, extantDir, hitCtx, weightBxDF, pdfBxDF);

					L *= renderCtx.emplacedRay.weight * albedo * 2.0f * weightBxDF; // Factor of two here accounts for stochastic dithering between direct and indirect sampling

					// If MIS is enabled, weight the ray using the power heuristic
					if (GetImportanceMode(renderCtx) == kImportanceMIS)
					{
						L *= PowerHeuristic(pdfLight, pdfBxDF);
					}

					renderCtx.EmplaceDirectSample(RayBasic(hitCtx.ExtantOrigin(), extantDir), L, pdfLight, lightIdx, kRayDirectLightSample);
				}
			}
			// Sample the BxDF
			else if (bxdf->Sample(incidentRay, hitCtx, renderCtx, extantDir, pdfBxDF))
			{	
				renderCtx.EmplaceDirectSample(RayBasic(hitCtx.ExtantOrigin(), extantDir), 
					renderCtx.emplacedRay.weight * albedo * 2.0f,
					pdfBxDF, lightIdx, kRayDirectBxDFSample);
			}	
		}

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
		
		//m_objects.cu_deviceAccumBuffer->At(renderCtx.viewportPos) = vec4(renderCtx.viewportPos.x / 512.0f, renderCtx.viewportPos.y / 512.0f, 0.0f, -1.0f);
		//return;

		compressedRay.Kill();

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
		
		vec3 L(0.0f);
		
		if (!hitObject)
		{
			// Ray didn't hit anything so add the ambient term multiplied by the weight
			L = m_params.ambientRadiance * compressedRay.weight;
		}
		else
		{		
			// Ray is a direct sample 
			if (incidentRay.IsDirectSample())
			{
				// Check that the intersected tracable is the same as the light ID associated with this ray
				if (compressedRay.lightId == hitObject->GetLightID())
				{
					// Light should be evaluated (i.e. BxDF was sampled)
					if (incidentRay.flags & kRayDirectBxDFSample)
					{
						const Light* light = (*m_objects.cu_deviceLights)[compressedRay.lightId];
						if (!light) { L = kPink; }
						else
						{
							float pdfLight;
							light->Evaluate(incidentRay, hitCtx, L, pdfLight);

							L *= compressedRay.weight;
							if (GetImportanceMode(renderCtx) == kImportanceMIS)
							{
								L *= PowerHeuristic(compressedRay.pdf, pdfLight);
							}
						}
					}
					// If the light itself was sampled, everything's baked into the throughput
					else
					{
						L = compressedRay.weight;
					}
				}
			}
			else if(hitObject->GetLightID() == kNotALight || GetImportanceMode(renderCtx) == kImportanceBxDF)
			{			
				// Otherwise, it's a BxDF sample so do a regular shade op
				L = Shade(incidentRay, *(hitObject->GetBoundMaterial()), hitCtx, renderCtx) * compressedRay.weight;
				//if(compressedRay.IsAlive()) L = compressedRay.od.d * 0.5f + vec3(0.5f);
			}	
		}

		// Accumulate radiance if we're above a certain threshold
		//if (cwiseMax(L) > 1e-6f)
		{
			m_objects.cu_deviceAccumBuffer->Accumulate(renderCtx.viewportPos, L, renderCtx.depth, compressedRay.IsAlive());
		}
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

		CompressedRay& compressedRay = (*m_objects.cu_deviceCompressedRayBuffer)[kKernelIdx];

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
			const auto type = object->GetAssetType();
			
			if (type == AssetType::kTracable)
			{
				Log::Debug("Linked tracable '%s' to wavefront tracer.\n", object->GetAssetID());

				Cuda::AssetHandle<Host::Tracable> tracable = object.DynamicCast<Tracable>();
				Assert(tracable);
				m_hostTracables->Push(tracable);
			}
			else if(type == AssetType::kLight)
			{
				Log::Debug("Linked light '%s' to wavefront tracer.\n", object->GetAssetID());

				Cuda::AssetHandle<Host::Light> light = object.DynamicCast<Light>();				
				Assert(light);				
			
				// Set the light ID for this tracable with the index of the light in the array. Crude, but it'll do for now. 
				Cuda::AssetHandle<Host::Tracable> tracable = light->GetTracableHandle();
				const uchar lightId = m_hostLights->Size();
				light->SetLightID(lightId);
				tracable->SetLightID(lightId);

				m_hostLights->Push(light);
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