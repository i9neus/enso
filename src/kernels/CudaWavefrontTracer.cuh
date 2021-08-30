#pragma once

#include "CudaRenderObject.cuh"
#include "CudaAssetContainer.cuh"
#include "math/CudaMath.cuh"

#include "cameras/CudaCamera.cuh"
#include "bxdfs/CudaLambert.cuh"
#include "materials/CudaSimpleMaterial.cuh"

namespace Json { class Node; }

namespace Cuda
{
	class HitCtx;
	class RenderCtx;
	class Ray;
	class CompressedRay;
	
	namespace Host { class WavefrontTracer; }	

	enum TracerPixelFlags : uchar 
	{ 
		kTracerPixelChanged = 1 
	};
	
	enum ImportanceMode : int 
	{ 
		kImportanceMIS, 
		kImportanceLight, 
		kImportanceBxDF 
	};

	enum LightSelectionMode : int 
	{ 
		kLightSelectionNaive, 
		kLightSelectionWeighted 
	};

	enum TraceMode : int 
	{ 
		kTraceWavefront, 
		kTracePath 
	};

	enum ShadingMode : int
	{
		kShadeFull,
		kShadeSimple,
		kShadeNormals,
		kShadeDebug
	};

	struct WavefrontTracerParams
	{
		__host__ __device__ WavefrontTracerParams();
		__host__ WavefrontTracerParams(const ::Json::Node& node, const uint flags) : WavefrontTracerParams() { FromJson(node, flags); }

		__host__ void ToJson(::Json::Node& node) const;
		__host__ void FromJson(const ::Json::Node& node, const uint flags);

		int			maxDepth;
		float		russianRouletteThreshold;
		int			importanceMode;
		int			traceMode;
		int			lightSelectionMode;
		int			shadingMode;
	};
	
	namespace Device
	{
		class Camera;
		class Material;
		class Tracable;
		class Light;
		class BxDF;
		template<typename T> class Array;
		
		using CompressedRayBuffer = Device::Array<CompressedRay>;
		using PixelFlagsBuffer = Device::Array<uchar>;

		class WavefrontTracer : public Device::Asset, public AssetTags<Host::WavefrontTracer, Device::WavefrontTracer>
		{		
			friend class Host::WavefrontTracer;
		public:
			struct Objects
			{
				Device::AssetContainer<Device::Tracable>*	cu_deviceTracables = nullptr;
				Device::AssetContainer<Device::Light>*		cu_deviceLights = nullptr;

				Device::Camera*								cu_camera = nullptr;			

				Device::CompressedRayBuffer*				cu_compressedRayBuffer = nullptr;
				Device::Array<uint>*						cu_blockRayOccupancy = nullptr;
				Device::RenderState::Stats*					cu_renderStats = nullptr;
			};		

		protected:			
			Objects							m_objects;
			WavefrontTracerParams			m_defaultParams;
			WavefrontTracerParams			m_activeParams;

			LambertBRDF						m_lightProbeBRDF;
			SimpleMaterial					m_lightProbeMaterial;

			float							m_wallTime;
			int								m_frameIdx;
			int								m_maxRayDepth;
			uint							m_checkDigit;
			int                             m_numLights;

			__device__ uchar GetImportanceMode(const RenderCtx& ctx) const;
			__device__ vec3 Shade(const Ray& incidentRay, const Device::Material& hitMaterial, const HitCtx& hitCtx, RenderCtx& renderCtx) const;
			__device__ void InitaliseScratchpadObjects() const;
			__device__ __forceinline__ bool ApplyRussianRoulette(float rayWeight, float& xi, float& outputWeight) const;

		public:
			__device__ WavefrontTracer();

			__device__ void Composite(const ivec2& viewportPos, Device::ImageRGBA* deviceOutputImage) const;
			__device__ bool Trace(CompressedRay& compressedRay) const;
			__device__ __forceinline__ void Trace(const uint rayIdx) const;
			__device__ __forceinline__ void TraceMultiple(const uint rayIdx) const;
			__device__ void PreFrame(const float& wallTime, const int frameIdx);
			__device__ void PreBlock() const;
			__device__ void Reduce();
			__device__ bool SelectLight(const Ray& incident, const HitCtx& hitCtx, const float& xi, int& lightIdx, float& weight) const;
			__device__ void SampleDirectComponent(const Ray& incidentRay, const HitCtx& hitCtx, RenderCtx& renderCtx, vec2& xi) const;

			__device__ void Synchronise(const Objects& objects);
			__device__ void Synchronise(const WavefrontTracerParams& params);

		};
	}

	namespace Host
	{
		class Material;
		class Tracable;
		class Light;
		class BxDF;
		class Camera;	
		
		class WavefrontTracer : public Host::RenderObject
		{
		private:
			Device::WavefrontTracer*							cu_deviceData;		
			Device::WavefrontTracer::Objects					m_deviceObjects;

			AssetHandle<Host::AssetContainer<Host::Tracable>>   m_hostTracables;
			AssetHandle<Host::AssetContainer<Host::Light>>      m_hostLights;

			AssetHandle<Host::Camera>							m_hostCameraAsset;
			AssetHandle<Host::CompressedRayBuffer>				m_hostCompressedRayBuffer;
			AssetHandle<Host::ImageRGBW>						m_hostAccumBuffer;

			WavefrontTracerParams								m_params;

			bool					m_isDirty;
			bool					m_isInitialised;
			std::string				m_cameraId;

		public:
			__host__ WavefrontTracer(const ::Json::Node& node, const std::string& id);
			__host__ virtual ~WavefrontTracer();

			__host__ static AssetHandle<Host::RenderObject>		Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);
			__host__ static std::string							GetAssetTypeString() { return "wavefront"; }
			__host__ static std::string							GetAssetDescriptionString() { return "Wavefront Tracer"; }
			__host__ virtual AssetType							GetAssetType() const override final { return AssetType::kIntegrator; }
			__host__ static AssetType							GetAssetStaticType() { return AssetType::kIntegrator; }
			
			__host__ virtual void								FromJson(const ::Json::Node& renderParamsJson, const uint flags) override final;
			__host__ virtual void								Bind(RenderObjectContainer& sceneObjects) override final;
			__host__ void										SetDirty() { m_isDirty = true; }

			__host__ virtual void								OnUpdateSceneGraph(RenderObjectContainer& sceneObjects) override final;
			__host__ virtual void								OnDestroyAsset() override final;
			__host__ virtual void								OnPreRenderPass(const float wallTime, const uint frameIdx) override final;
			__host__ const WavefrontTracerParams&				GetParams() const { return m_params; }

			__host__ void										Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage);
			__host__ void										Trace(); 
			__host__ AssetHandle<Host::Camera>					GetAttachedCamera() { return m_hostCameraAsset; }
			__host__ void										AttachCamera(AssetHandle<Host::Camera>& camera);
		};
	}
}