#include "CudaWavefrontTracer.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"
#include "generic/Assert.h"
#include "CudaAsset.cuh"
#include "CudaRay.cuh" 

#include <thrust/sort.h>

#include "bxdfs/CudaBxDF.cuh"
#include "materials/CudaMaterial.cuh"
#include "lights/CudaLight.cuh"
#include "tracables/CudaTracable.cuh"
#include "cameras/CudaCamera.cuh"
#include "math/CudaColourUtils.cuh"

#include "CudaManagedArray.cuh"

#include "CudaCommonIncludes.cuh"
#include "CudaImage.cuh"
#include "CudaCtx.cuh"

#include "generic/JsonUtils.h"

#define kMaxLights 10

namespace Cuda
{
	__host__ WavefrontTracerParams::WavefrontTracerParams() :
		maxDepth(1),
		ambientRadiance(0.0f),
		shadingMode(kShadeFull),
		importanceMode(kImportanceMIS),
		lightSelectionMode(kLightSelectionNaive),
		traceMode(kTracePath)
	{
	}

	__host__ void WavefrontTracerParams::ToJson(::Json::Node& node) const
	{
		node.AddValue("maxDepth", maxDepth); 
		node.AddArray("ambientRadiance", std::vector<float>({ ambientRadiance.x, ambientRadiance.y, ambientRadiance.z }));
		node.AddEnumeratedParameter("importanceMode", std::vector<std::string>({ "mis", "light", "bxdf" }), importanceMode);
		node.AddEnumeratedParameter("traceMode", std::vector<std::string>({ "wavefront", "path" }), traceMode);
		node.AddEnumeratedParameter("lightSelectionMode", std::vector<std::string>({ "naive", "weighted" }), lightSelectionMode);
		node.AddEnumeratedParameter("shadingMode", std::vector<std::string>({ "full", "simple", "normals", "debug" }), shadingMode);
	}

	__host__ void WavefrontTracerParams::FromJson(const ::Json::Node& node, const uint flags)
	{
		node.GetValue("maxDepth", maxDepth, flags);
		node.GetVector("ambientRadiance", ambientRadiance, ::Json::kSilent);
		node.GetEnumeratedParameter("importanceMode", std::vector<std::string>({ "mis", "light", "bxdf" }), importanceMode, flags);
		node.GetEnumeratedParameter("traceMode", std::vector<std::string>({ "wavefront", "path" }), traceMode, flags);
		node.GetEnumeratedParameter("lightSelectionMode", std::vector<std::string>({ "naive", "weighted" }), lightSelectionMode, flags);
		node.GetEnumeratedParameter("shadingMode", std::vector<std::string>({ "full", "simple", "normals", "debug" }), shadingMode, flags);
	}

	__host__ bool WavefrontTracerParams::operator==(const WavefrontTracerParams& rhs) const
	{
		return maxDepth == rhs.maxDepth &&
			ambientRadiance == rhs.ambientRadiance;
	}

	__device__ Device::WavefrontTracer::WavefrontTracer() : m_checkDigit(0)
	{
	}

	__device__ void Device::WavefrontTracer::Synchronise(const Device::WavefrontTracer::Objects& objects)
	{
		m_objects = objects;
		m_activeParams = m_defaultParams;

		// If a camera is attached then load the objects into the block
		if (objects.cu_camera)
		{
			const auto& renderState = objects.cu_camera->GetRenderState();
			m_objects.cu_compressedRayBuffer = renderState.cu_compressedRayBuffer;
			m_objects.cu_blockRayOccupancy = renderState.cu_blockRayOccupancy;
			m_objects.cu_renderStats = renderState.cu_renderStats;

			assert(m_objects.cu_compressedRayBuffer);
			assert(m_objects.cu_blockRayOccupancy);
			assert(m_objects.cu_renderStats);
		
			// Override the parameters as necessary
			const auto& cameraParams = m_objects.cu_camera->GetParams();
			if (cameraParams.overrides.maxDepth > -1) { m_activeParams.maxDepth = cameraParams.overrides.maxDepth; }
		}

		m_numLights = m_objects.cu_deviceLights->Size();
		assert(m_numLights <= kMaxLights);
	}

	__device__ void Device::WavefrontTracer::Synchronise(const WavefrontTracerParams& params)
	{
		// We keep two copies of these parameters because attached cameras may need to override them
		m_defaultParams = params;
		m_activeParams = params;
	}

	__device__ void Device::WavefrontTracer::PreFrame(const float& wallTime, const int frameIdx)
	{
		m_wallTime = wallTime;
		m_frameIdx = frameIdx;
	}

	__device__ __forceinline__ float PowerHeuristic(float pdf1, float pdf2)
	{
		return 2.0f * sqr(pdf1) / (sqr(pdf1) + sqr(pdf2));
	}

	__device__ uchar Device::WavefrontTracer::GetImportanceMode(const RenderCtx& ctx) const
	{
		return m_activeParams.importanceMode;
		//return (ctx.emplacedRay.viewport.x < 256) ? m_activeParams.importanceMode : kImportanceMIS;
	}

	template<typename T>
	__device__ __inline__ int LowerBound(int i0, int i1, const T* pmf, const float& xi)
	{
		while (i1 - i0 > 1)
		{
			const int iMid = i0 + (i1 - i0) / 2;
			if (pmf[iMid] < xi) { i0 = iMid; }
			else { i1 = iMid; }
		}

		if (pmf[i1] < xi) { return i1 + 1; }
		else if (pmf[i0] < xi) { return i0 + 1; }
		return i0;
	}

	__device__ bool Device::WavefrontTracer::SelectLight(const Ray& incident, const HitCtx& hitCtx, const float& xi, int& lightIdx, float& weight) const
	{		
		float pmf[kMaxLights + 1];
		pmf[0] = 0.0f;
		for (int idx = 0; idx < m_numLights && idx < kMaxLights; ++idx)
		{
			float estimate = (*m_objects.cu_deviceLights)[idx]->Estimate(incident, hitCtx);
			
			pmf[1 + idx] = pmf[idx] + estimate;
		}

		// No lights are in range to sample
		if (pmf[m_numLights] == 0.0f) { return false; }

		lightIdx = LowerBound(1, m_numLights, pmf, xi * pmf[m_numLights]) - 1;
		weight = pmf[m_numLights] / (pmf[lightIdx + 1] - pmf[lightIdx]);

		assert(lightIdx >= 0 && lightIdx < m_numLights);

		return true;
	}

	__device__ vec3 Device::WavefrontTracer::Shade(const Ray& incidentRay, const Device::Material& material, const HitCtx& hitCtx, RenderCtx& renderCtx) const
	{
		vec3 albedo;
		vec3 L(0.0f);
		if (!hitCtx.backfacing)
		{
			material.Evaluate(hitCtx, albedo, L);
		}

		if (m_activeParams.shadingMode == kShadeSimple) { return albedo; }
	
		const BxDF* bxdf = material.GetBoundBxDF();
		if (!bxdf) { return L; }
		
		// Only shade back-facing surfaces if this BxDF supports it
		if (hitCtx.backfacing && !bxdf->IsTwoSided()) { return kZero; }

		// If this isn't a probe ray, add in the cached radiance
		if (!(incidentRay.flags & kRayLightProbe))
		{
			L += bxdf->EvaluateCachedRadiance(hitCtx) * albedo;
		}

		// If we're beyond max depth, just return incandescence
		if (renderCtx.depth > m_activeParams.maxDepth) { return L; }

		// Generate some random numbers
		vec2 xi = renderCtx.rng.Rand<2, 3>();
		// Weight to compensate for stochastic branching
		float stochasticWeight = 2.0f;

		// If we're at max depth, only do NEE 
		if (renderCtx.depth == m_activeParams.maxDepth)
		{
			if (m_numLights == 0) { return L; }
			xi = xi * 0.5f + 0.5f;
			stochasticWeight = 1.0f;
		}

		// If there are no lights in this scene, always sample the BxDF
		else if (GetImportanceMode(renderCtx) == kImportanceBxDF || m_numLights == 0)
		{
			xi.x *= 0.5f;
			stochasticWeight = 1.0f;
		}

		// Indirect light sampling
		if (xi.x < 0.5f)
		{
			vec3 extantDir;
			float pdfBxDF;
			if (bxdf->Sample(incidentRay, hitCtx, renderCtx, extantDir, pdfBxDF))
			{
				vec3 LIndirect = renderCtx.emplacedRay.weight* albedo;
				LIndirect *= stochasticWeight;

				renderCtx.EmplaceIndirectSample(RayBasic(hitCtx.ExtantOrigin(), extantDir), LIndirect, kRayScattered);
			}
		}
		// Direct light sampling
		else
		{
			// Rescale the random number
			xi.x = (GetImportanceMode(renderCtx) == kImportanceLight) ? 0.0f : (xi.x * 2.0f - 1.0f);

			// Select a light to sample or evaluate
			int lightIdx = 0;
			float weightLight = 1.0f;
			if (m_numLights > 1)
			{
				// Weight each light equally. Cheap to sample but noisy for scenes with lots of lights.
				if (m_activeParams.lightSelectionMode == kLightSelectionNaive)
					//if (renderCtx.emplacedRay.GetViewportPos().x < 256)
				{
					lightIdx = min(m_numLights - 1, int(xi.y * m_numLights));
					weightLight = m_numLights;
				}
				// Build a PMF based on a crude estiamte of the irradiance at the shading point. Expensive, but 
				// significantly reduces noise. 
				else if (!SelectLight(incidentRay, hitCtx, xi.y, lightIdx, weightLight))
				{
					return L;
				}
			}

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

					L *= renderCtx.emplacedRay.weight * albedo * stochasticWeight * weightBxDF * weightLight; // Factor of two here accounts for stochastic dithering between direct and indirect sampling

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
					renderCtx.emplacedRay.weight * albedo * stochasticWeight,
					pdfBxDF, lightIdx, kRayDirectBxDFSample);
			}
		}

		return L;
	}

	__device__ void Device::WavefrontTracer::PreBlock() const
	{
		auto I = m_objects.cu_deviceTracables->Size();
		for (int i = 0; i < I; i++)
		{
			(*m_objects.cu_deviceTracables)[i]->InitialiseKernelConstantData();
		}
	}

	__device__ __forceinline__ void Device::WavefrontTracer::TraceMultiple(const uint rayIdx) const
	{
		assert(m_objects.cu_compressedRayBuffer);
		if (rayIdx >= m_objects.cu_compressedRayBuffer->Size()) { return; }

		if (kThreadIdx == 0) { PreBlock(); }
		__syncthreads();

		CompressedRay& compressedRay = (*m_objects.cu_compressedRayBuffer)[rayIdx];
		bool isNextRay;
		do
		{			
			isNextRay = Trace(compressedRay);
		} 
		while (isNextRay);
	}

	__device__ __forceinline__ void Device::WavefrontTracer::Trace(const uint rayIdx) const
	{
		assert(m_objects.cu_compressedRayBuffer);
		if (rayIdx >= m_objects.cu_compressedRayBuffer->Size()) { return; }

		if (kThreadIdx == 0) { PreBlock(); }
		__syncthreads();

		Trace((*m_objects.cu_compressedRayBuffer)[rayIdx]);
	}

	__device__ bool Device::WavefrontTracer::Trace(CompressedRay& compressedRay) const
	{
		if (!compressedRay.IsAlive()) { return false; }
		
		__shared__ uchar deadRays[16 * 16];

		Ray incidentRay(compressedRay);
		RenderCtx renderCtx(compressedRay);
		vec3 L(0.0f);

		//m_objects.cu_deviceAccumBuffer->At(renderCtx.viewportPos) = vec4(renderCtx.viewportPos.x / 512.0f, renderCtx.viewportPos.y / 512.0f, 0.0f, -1.0f);
		//return;

		compressedRay.Kill();

		// INTERSECTION
		HitCtx hitCtx;
		hitCtx.debug = 0.0f;
		auto& tracables = *m_objects.cu_deviceTracables;
		Device::Tracable* hitObject = nullptr;
		const int numTracables = tracables.Size();
		for (int i = 0; i < numTracables; i++)
		{
			if (tracables[i]->Intersect(incidentRay, hitCtx))
			{
				hitObject = tracables[i];
			}
		}

		if (m_activeParams.shadingMode == kShadeNormals)
		{
			if (hitObject)
			{
				L = hitCtx.hit.n * 0.5f + vec3(0.5f);
			}
			m_objects.cu_camera->Accumulate(renderCtx, hitCtx, L);
			return false;
		}

		if (!hitObject)
		{
			if (incidentRay.IsIndirectSample())
			{
				// Ray didn't hit anything so add the ambient term multiplied by the weight
				L = m_activeParams.ambientRadiance * compressedRay.weight;
			}
		}
		else
		{	
			if (m_activeParams.shadingMode == kShadeSimple)
			{
				const Device::Material* boundMaterial = hitObject->GetBoundMaterial();
				if (boundMaterial)
				{
					vec3 albedo;
					boundMaterial->Evaluate(hitCtx, albedo, L);
					m_objects.cu_camera->Accumulate(renderCtx, hitCtx, albedo * -dot(incidentRay.od.d, hitCtx.hit.n));
				}
				return false;
			}
			
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
							if (light->Evaluate(incidentRay, hitCtx, L, pdfLight))
							{
								L *= compressedRay.weight;
								L *= float(m_objects.cu_deviceLights->Size());
								if (GetImportanceMode(renderCtx) == kImportanceMIS)
								{
									L *= PowerHeuristic(compressedRay.pdf, pdfLight);
								}
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
			else if (hitObject->GetLightID() == kNotALight || GetImportanceMode(renderCtx) == kImportanceBxDF || incidentRay.flags == kRaySpecular)
			{
				const Device::Material* boundMaterial = hitObject->GetBoundMaterial();
				if (boundMaterial)
				{
					// Otherwise, it's a BxDF sample so do a regular shade op
					L = Shade(incidentRay, *boundMaterial, hitCtx, renderCtx) * compressedRay.weight;
				}
				else { L += kPink;	}
			}
		}

		if (m_activeParams.shadingMode == kShadeDebug)
		{
			L = hitCtx.debug;
			compressedRay.Kill();
		}

		m_objects.cu_camera->Accumulate(renderCtx, hitCtx, L);

		deadRays[kThreadIdx] = !compressedRay.IsAlive();

		// Reduce the contents of the block buffer to count the number of dead rays 
		__syncthreads();
		for (int i = 128; i >= 8; i >>= 1)
		{
			if (kThreadIdx < i)
			{
				deadRays[kThreadIdx] += deadRays[kThreadIdx + i];
			}
			__syncthreads();
		}

		if (kThreadIdx < 8)
		{
			(*m_objects.cu_blockRayOccupancy)[kBlockIdx * 8 + kThreadIdx] = deadRays[0];
		}

		return compressedRay.IsAlive();
	}

	__device__ void Device::WavefrontTracer::Reduce()
	{
		auto& occupancyBuffer = *m_objects.cu_blockRayOccupancy;

		for (uint i = 4096; i > 0; i >>= 1)
		{
			if (kKernelIdx < i)
			{
				occupancyBuffer[kKernelIdx] += occupancyBuffer[kKernelIdx + i];
			}
			__syncthreads();
		}

		m_objects.cu_renderStats->deadRays = occupancyBuffer[0];
	}	

	__host__ void Host::WavefrontTracer::OnDestroyAsset()
	{
		m_hostTracables.DestroyAsset();
		m_hostLights.DestroyAsset();

		DestroyOnDevice(cu_deviceData);
	}

	__host__ AssetHandle<Host::RenderObject> Host::WavefrontTracer::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
	{
		if (expectedType != AssetType::kIntegrator) { return AssetHandle<Host::RenderObject>(); }

		return AssetHandle<Host::RenderObject>(new Host::WavefrontTracer(json, id), id);
	}

	__host__ Host::WavefrontTracer::WavefrontTracer(const ::Json::Node& node, const std::string& id) :
		cu_deviceData(nullptr),
		m_isDirty(true),
		m_isInitialised(true)
	{
		m_hostTracables = AssetHandle<Host::AssetContainer<Host::Tracable>>(tfm::format("%s_tracablesContainer", id));
		m_hostLights = AssetHandle<Host::AssetContainer<Host::Light>>(tfm::format("%s_lightsContainer", id));

		cu_deviceData = InstantiateOnDevice<Device::WavefrontTracer>();
		FromJson(node, ::Json::kRequiredWarn);
	}

	__host__ Host::WavefrontTracer::~WavefrontTracer()
	{
		OnDestroyAsset();
	}

	__host__ void Host::WavefrontTracer::Bind(RenderObjectContainer& sceneObjects)
	{		
		Log::Indent indent;

		// Do some clean-up...
		m_hostTracables->Clear();
		m_hostLights->Clear();
		
		// Loop over the scene objects and look for tracables and lights
		for (const auto& object : sceneObjects)
		{
			// Don't push objects belonging to other objects. We'll do this explicitly later.
			if (object->IsChildObject()) { continue; }
			
			// If the object is flagged as disabled, exclude it.
			const RenderObjectParams* objectParams = object->GetRenderObjectParams();
			if (objectParams && objectParams->flags() & kRenderObjectDisabled) { continue; }
			
			const auto type = object->GetAssetType();
			if (type == AssetType::kTracable)
			{
				Log::Debug("Linked tracable '%s' to wavefront tracer.\n", object->GetAssetID());

				Cuda::AssetHandle<Host::Tracable> tracable = object.DynamicCast<Tracable>();
				Assert(tracable);
				m_hostTracables->Push(tracable);
			}
			else if (type == AssetType::kLight)
			{
				Log::Debug("Linked light '%s' to wavefront tracer.\n", object->GetAssetID());

				Cuda::AssetHandle<Host::Light> light = object.DynamicCast<Light>();
				Assert(light);

				// Set the light ID for this tracable with the index of the light in the array. Crude, but it'll do for now. 
				Cuda::AssetHandle<Host::Tracable> tracable = light->GetTracableHandle();
				const uchar lightId = m_hostLights->Size();
				light->SetLightID(lightId);
				tracable->SetLightID(lightId);

				// Push both the light and tracable into the list
				m_hostLights->Push(light);
				m_hostTracables->Push(tracable);
			}
		}
		
		// Sort the tracables by intersection cost
		auto functor = [](const AssetHandle<Host::Tracable>& lhs, const AssetHandle<Host::Tracable>& rhs)
		{
			return lhs->GetIntersectionCostHeuristic() < rhs->GetIntersectionCostHeuristic();
		};
		m_hostTracables->SetSortFunctor(functor);		

		// Synchronise the container objects managed by this instance
		m_hostTracables->Synchronise();
		m_hostLights->Synchronise();

		// Synchronise the wavefront tracer structure on the device
		m_deviceObjects.cu_deviceTracables = m_hostTracables->GetDeviceInstance();
		m_deviceObjects.cu_deviceLights = m_hostLights->GetDeviceInstance();

		SynchroniseObjects(cu_deviceData, m_deviceObjects);

		if (!m_isInitialised)
		{
			Log::Write("Bound tracables and lights to wavefront tracer '%s'.\n", GetAssetID());
		}
	}

	__host__ void Host::WavefrontTracer::OnUpdateSceneGraph(RenderObjectContainer& sceneObjects)
	{
		// Do a complete re-bind when the scene graph updates
		Bind(sceneObjects);
	}

	__host__ void Host::WavefrontTracer::FromJson(const ::Json::Node& parentNode, const uint flags)
	{
		Host::RenderObject::UpdateDAGPath(parentNode);

		m_params.FromJson(parentNode, flags);
		SynchroniseObjects(cu_deviceData, m_params);

		parentNode.GetValue("camera", m_cameraId, flags);
		m_isDirty = true;
	}

	__global__ void KernelPreFrame(Device::WavefrontTracer* tracer, const float wallTime, const int frameIdx)
	{
		tracer->PreFrame(wallTime, frameIdx);
	}

	__global__ void KernelTrace(Device::WavefrontTracer* tracer)
	{	
		tracer->Trace(kKernelIdx);
	}

	__global__ void KernelTraceMultiple(Device::WavefrontTracer* tracer)
	{
		tracer->TraceMultiple(kKernelIdx);
	}

	__global__ void KernelReduce(Device::WavefrontTracer* tracer)
	{
		tracer->Reduce();
	}

	__host__ void Host::WavefrontTracer::AttachCamera(AssetHandle<Host::Camera>& camera)
	{
		m_hostCameraAsset = camera;
		if (!m_hostCameraAsset) { return; }		

		// Synchronise the attached camera with the device
		m_deviceObjects.cu_camera = m_hostCameraAsset->GetDeviceInstance();
		SynchroniseObjects(cu_deviceData, m_deviceObjects);

		m_hostCompressedRayBuffer = m_hostCameraAsset->GetCompressedRayBuffer();
		m_hostAccumBuffer = m_hostCameraAsset->GetAccumulationBuffer();
	}

	__host__ void Host::WavefrontTracer::Trace()
	{
		if (!m_isInitialised || !m_hostCameraAsset) { return; }		

		switch (m_params.traceMode)
		{
		case kTraceWavefront:
			KernelTrace << <  m_hostCompressedRayBuffer->GetGridSize(), m_hostCompressedRayBuffer->GetBlockSize(), 0, m_hostStream >> > (cu_deviceData);
			break;
		case kTracePath:
			KernelTraceMultiple << <  m_hostCompressedRayBuffer->GetGridSize(), m_hostCompressedRayBuffer->GetBlockSize(), 0, m_hostStream >> > (cu_deviceData);
			break;
		}

		KernelReduce << < 8192 / 256, 256, 0, m_hostStream >> > (cu_deviceData);
	}

	__host__ void Host::WavefrontTracer::OnPreRenderPass(const float wallTime, const uint frameIdx)
	{
		if (!m_isInitialised || !m_hostCameraAsset) { return; }

		KernelPreFrame << < 1, 1, 0, m_hostStream >> > (cu_deviceData, wallTime, frameIdx);
	}
}