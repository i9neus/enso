﻿#include "../CudaSampler.cuh"
#include "../CudaHash.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRay.cuh"

#include "CudaPerspectiveCamera.cuh"
#include "generic/JsonUtils.h"
#include "../CudaManagedArray.cuh"
#include "../CudaManagedObject.cuh"

#define kRayBufferWidth         512u
#define kRayBufferHeight        512u
#define kRayBufferSize          (kRayBufferWidth * kRayBufferHeight * 2)

#define kCameraAA                 1.5f             // The width/height of the anti-aliasing kernel in pixels
#define kCameraSensorSize         0.035f           // The size of the camera sensor in meters
#define kBlades                   5.0f
#define kBladeCurvature           0.0f
#define kCameraUp                 vec3(0.0f, 1.0f, 0.0f) 

namespace Cuda
{    
    __host__ __device__ PerspectiveCameraParams::PerspectiveCameraParams()
    {
        position = vec3(0.5f, 1.0f, 1.5f);
        lookAt = vec3(0.0f, 0.5f, 0.0f);
        focalPlane = 1.0f;
        fLength = 0.45f;
        fStop = 0.45f;
        displayExposure = 0.0f;
        displayGamma = 1.0f;
        viewportDims = ivec2(512, 512);
        isRealtime = false;
        lightProbeEmulation = kLightProbeEmulationNone;
    }

    __host__ PerspectiveCameraParams::PerspectiveCameraParams(const ::Json::Node& node) :
        PerspectiveCameraParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void PerspectiveCameraParams::ToJson(::Json::Node& node) const
    {
        camera.ToJson(node);

        node.AddArray("pos", std::vector<float>({ position.x, position.y, position.z }));
        node.AddArray("lookAt", std::vector<float>({ lookAt.x, lookAt.y, lookAt.z }));
        node.AddValue("focalPlane", focalPlane);
        node.AddValue("fLength", fLength);
        node.AddValue("fStop", fStop);
        node.AddValue("displayExposure", displayExposure);
        node.AddValue("displayGamma", displayGamma);
        node.AddValue("isRealtime", isRealtime);
        node.AddEnumeratedParameter("lightProbeEmulation", std::vector<std::string>({ "none", "all", "direct", "indirect" }), lightProbeEmulation);
    }

    __host__ void PerspectiveCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        camera.FromJson(node, flags);

        node.GetVector("pos", position, flags);
        node.GetVector("lookAt", lookAt, flags);
        node.GetValue("focalPlane", focalPlane, flags);
        node.GetValue("fLength", fLength, flags);
        node.GetValue("fStop", fStop, flags);
        node.GetValue("displayExposure", displayExposure, flags);
        node.GetValue("displayGamma", displayGamma, flags);
        node.GetValue("isRealtime", isRealtime, flags);
        node.GetEnumeratedParameter("lightProbeEmulation", std::vector<std::string>({ "none", "all", "direct", "indirect" }), lightProbeEmulation, flags);
    }

    // Returns the polar distance r to the perimeter of an n-sided polygon
    __device__ __forceinline__ float Ngon(float phi)
    {
        float piBlades = kPi / kBlades;
        float bladeRadius = cos(piBlades) / cos(fmodf(((phi)+piBlades) + piBlades, 2.0f * piBlades) - piBlades);

        // Take into account the blade curvature
        return mix(bladeRadius, 1.0f, kBladeCurvature);
    }

    __device__ Device::PerspectiveCamera::PerspectiveCamera()
    {
        Prepare();
    }

    __device__ void Device::PerspectiveCamera::Composite(const ivec2& accumPos, Device::ImageRGBA* deviceOutputImage) const
    {        
        const ivec2 viewportPos = accumPos + deviceOutputImage->Dimensions() / 2 - m_objects.cu_accumBuffer->Dimensions() / 2;        
        if (viewportPos.x < 0 || viewportPos.x >= deviceOutputImage->Width() || 
            viewportPos.y < 0 || viewportPos.y >= deviceOutputImage->Height()) { return; }

        // If the texel weight is negative, the texel is ready to be rendered
        vec4& texel = m_objects.cu_accumBuffer->At(accumPos);
        if (texel.w >= 0.0f) { return; }

        // Flip the weight back to positve
        texel.w = -texel.w;
        // Normalise
        vec3 rgb = texel.xyz / fmax(1.0f, texel.w);
        // Apply exposure/gamma correction
        rgb = pow(rgb * m_displayExposure, vec3(1.0f / m_params.displayGamma));

        deviceOutputImage->At(viewportPos) = vec4(rgb, 1.0f);
    }

    __device__ void Device::PerspectiveCamera::Prepare()
    { 
        // Only use the lower 31 bits for the seed because we need to deduce the actual sample count from it
        m_seedOffset = HashOf(m_params.camera.seed) & ((1 << 31) - 1);
        
        //float theta = kTwoPi * (m_cameraPos.x - 0.5f);
        //vec3 cameraPos = vec3(cos(theta), m_cameraPos.y, sin(theta)) * 5.0f * powf(2.0f, mix(-5.0f, 1.0f, m_cameraFLength.x));
        m_cameraPos = m_params.position;

        //float cameraOriginDist = length(cameraPos);
        //vec3 cameraLookAt = vec3(-10.0f * (m_cameraLook - vec2(0.5f)), cameraOriginDist);
        //cameraLookAt = createBasis(-cameraPos / cameraOriginDist, kCameraUp) * cameraLookAt;
        vec3 cameraLookAt = m_params.lookAt;
        
        vec3 cameraForward = cameraLookAt - m_cameraPos;
        m_focalDistance = length(cameraForward);
        cameraForward /= m_focalDistance;

        m_basis = CreateBasis(cameraForward, kCameraUp);

        m_focalDistance *= m_params.focalPlane;
        m_fStop = powf(2.0f, mix(-3.0, 8.0, m_params.fStop));

        // Define the focal length and F-number depending, either from built-in or user-defined values
        m_focalLength = powf(2.0f, mix(-9.0f, -0.5f, m_params.fLength));

        // Solve the thin-lens equation. http://hyperphysics.phy-astr.gsu.edu/hbase/geoopt/lenseq.html
        m_d1 = 0.5 * (m_focalDistance - sqrt(-4.0 * m_focalLength * m_focalDistance + sqr(m_focalDistance)));
        m_d2 = m_focalDistance - m_d1;

        m_displayExposure = powf(2.0f, m_params.displayExposure);
    }

    __device__ void Device::PerspectiveCamera::SeedRayBuffer(const ivec2& viewportPos, const uint frameIdx)
    {
        assert(kKernelIdx * 2 < kRayBufferSize);

        CompressedRay* compressedRays = &(*m_objects.renderState.cu_compressedRayBuffer)[(viewportPos.y * kRayBufferWidth + viewportPos.x) * 2];

        if (frameIdx == 0)
        {
            compressedRays[0].Reset();
            compressedRays[1].Reset();
            compressedRays[0].sampleIdx = m_seedOffset;
        }

        if (!compressedRays[0].IsAlive() && !compressedRays[1].IsAlive() &&
            (m_params.camera.minMaxSamples.y <= 0 || int(compressedRays[0].sampleIdx - m_seedOffset) < m_params.camera.minMaxSamples.y))
        {
            
            if (m_params.isRealtime)
            {
                m_objects.cu_accumBuffer->At(compressedRays[0].GetViewportPos()) = vec4(0.0f);
            }

            // Create the camera ray index 0
            CreateRay(viewportPos, compressedRays[0]);
        }
    }

    
    __device__ void Device::PerspectiveCamera::CreateRay(const ivec2& viewportPos, CompressedRay& ray) const
    {         
        __shared__ bool isInited;
        __shared__ mat3 basis;
        __shared__ vec3 cameraPos;
        __shared__ float focalDistance, focalLength, fStop, d1, d2;
        isInited = false;

        __syncthreads();
        
        // Load the camera data from global to shared memory
        if (!isInited)
        {
            basis = m_basis;
            cameraPos = m_cameraPos;
            d1 = m_d1;
            d2 = m_d2;
            focalLength = m_focalLength;
            focalDistance = m_focalDistance;
            fStop = m_fStop;
            isInited = true;
        }

        __syncthreads();

        // Update the ray with the new properties and generate a random sampler from it
        ray.SetViewportPos(viewportPos);
        ray.sampleIdx++;
        ray.depth = 0;
        RNG rng(ray);

        // Generate 4 random numbers from a continuous uniform distribution
        vec4 xi = rng.Rand<0, 1, 2, 3>();

        // The value of mu is used to sample the spectral wavelength but also the chromatic aberration effect.
        // If we're using the Halton low-disrepancy sampler, hash the input values and sample the sequence
        float mu = xi.y;

        /*float theta = kTwoPi * (m_cameraPos.x - 0.5f);
        //vec3 cameraPos = vec3(cos(theta), m_cameraPos.y, sin(theta)) * 5.0f * powf(2.0f, mix(-5.0f, 1.0f, m_cameraFLength.x));
        cameraPos = vec3(1.0f, 1.5f, 3.0f);

        float cameraOriginDist = length(cameraPos);
        vec3 cameraLookAt = vec3(-10.0f * (m_params.cameraLook - vec2(0.5f)), cameraOriginDist);
        //cameraLookAt = createBasis(-cameraPos / cameraOriginDist, kCameraUp) * cameraLookAt;
        cameraLookAt = vec3(0.0f);

        vec3 cameraForward = cameraLookAt - cameraPos;
        focalDistance = length(cameraForward);
        cameraForward /= focalDistance;

        focalDistance *= mix(0.0f, 1.0f, m_params.cameraFStop.y);
        fStop = powf(2.0f, mix(-3.0, 8.0, m_params.cameraFStop.x));

        basis = CreateBasis(cameraForward, kCameraUp);

        // Define the focal length and F-number depending, either from built-in or user-defined values
        focalLength = powf(2.0f, mix(-9.0f, -0.5f, m_params.cameraFLength.y));

        // Solve the thin-lens equation. http://hyperphysics.phy-astr.gsu.edu/hbase/geoopt/lenseq.html
        d1 = 0.5 * (focalDistance - sqrt(-4.0 * focalLength * focalDistance + sqr(focalDistance)));
        d2 = focalDistance - d1; */

        // Generate a position on the sensor, the focal plane, and the lens. This lens will always have circular bokeh
        // but with a few minor additions it's possible to add custom shapes such as irises. We reuse some random numbers
        // but this doesn't really matter because the anti-aliasing kernel is so small that we won't notice the correlation 
        // FIXME: Do an automatic cast
        vec2 sensorPos = vec2(xi.x, xi.y) * kCameraSensorSize * kCameraAA / max(m_params.viewportDims.x, m_params.viewportDims.y) +
            kCameraSensorSize * (vec2(viewportPos) - vec2(m_params.viewportDims) * 0.5) / float(max(m_params.viewportDims.x, m_params.viewportDims.y));
        vec2 focalPlanePos = vec2(sensorPos.x, sensorPos.y) * d2 / d1;

        vec2 lensPos;
        for (int i = 0; i < 10; i++)
        {
            lensPos = xi.zw * 2.0 - vec2(1.0);
            float bladeDist = Ngon(atan2f(lensPos.y, lensPos.x) + kPi);
            //if (length2(lensPos) < sqr(bladeDist)) { break; }
            //xi = renderCtx.pcg();
        }
        lensPos *= 0.5 * focalLength / fStop;
        //vec2 lensPos = sampleUnitDisc(xi.xy) * 0.5 * focalLength / fStop;

        // Assemble the primary
        ray.od.o = basis * vec3(lensPos, d1);
        ray.od.d = normalize((basis * vec3(focalPlanePos, focalDistance)) - ray.od.o);
        ray.od.o += cameraPos;
        ray.weight = 1.0f;
        ray.depth = 1;
        ray.flags = kRaySpecular;

        if (m_params.lightProbeEmulation != kLightProbeEmulationNone)
        {
            ray.flags = kRayIndirectSample | kRayLightProbe;
        }

        //ray.lambda = mix(3800.0f, 7000.0f, mu);
    }

    __device__ void Device::PerspectiveCamera::Accumulate(const RenderCtx& ctx, const Ray& incidentRay, const HitCtx& hitCtx, const vec3& value, const bool isAlive)
    {
        // Ray isn't carrying any energy and its weight is zero so bail out early.
        if (cwiseMax(value) < 1e-15f && isAlive) { return; }
        
        bool accumulate = m_params.lightProbeEmulation <= kLightProbeEmulationAll ||
                         (m_params.lightProbeEmulation == kLightProbeEmulationDirect && incidentRay.depth == 1) ||
                         (m_params.lightProbeEmulation == kLightProbeEmulationIndirect && incidentRay.depth > 1);

        if (incidentRay.depth < m_params.camera.overrides.minDepth) { accumulate = false; }

        vec3 L(0.0f);
        if (accumulate)
        {
            L = value;
            if (m_params.camera.splatClamp > 0.0)
            {
                const float intensity = cwiseMax(L);
                if (intensity > m_params.camera.splatClamp)
                {
                    L *= m_params.camera.splatClamp / intensity;
                }
            }
        }
        
        m_objects.cu_accumBuffer->Accumulate(ctx.emplacedRay[0].GetViewportPos(), L, isAlive);
    }

    __host__ Host::PerspectiveCamera::PerspectiveCamera(const std::string& id, const ::Json::Node& parentNode) :
        Host::Camera(parentNode, id, kRayBufferSize)
    {
        // Create the accumulation buffer
        m_hostAccumBuffer = CreateAsset<Host::ImageRGBW>(tfm::format("%s_perspAccumBuffer", id), 512, 512, m_hostStream);
        m_hostAccumBuffer->Clear(vec4(0.0f));
        
        // Instantiate the camera object on the device
        cu_deviceData = InstantiateOnDevice<Device::PerspectiveCamera>(id);

        FromJson(parentNode, ::Json::kRequiredWarn);

        // Sychronise the device objects
        Device::PerspectiveCamera::Objects deviceObjects;
        deviceObjects.cu_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        deviceObjects.renderState.cu_compressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
        deviceObjects.renderState.cu_blockRayOccupancy = m_hostBlockRayOccupancy->GetDeviceInstance();
        deviceObjects.renderState.cu_renderStats = m_hostRenderStats->GetDeviceInstance();
        SynchroniseObjects(cu_deviceData, deviceObjects);

        m_blockSize = dim3(16, 16, 1);
        m_gridSize = dim3((m_params.viewportDims.x + 15) / 16, (m_params.viewportDims.y + 15) / 16, 1);
    }

    __host__ AssetHandle<Host::RenderObject> Host::PerspectiveCamera::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kCamera) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::PerspectiveCamera>(id, json);
    }

    __host__ void Host::PerspectiveCamera::OnDestroyAsset()
    {
        Host::Camera::OnDestroyAsset();
        
        m_hostAccumBuffer.DestroyAsset();
        
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __host__ void Host::PerspectiveCamera::FromJson(const ::Json::Node& parentNode, const uint flags)
    {        
        m_params = PerspectiveCameraParams(parentNode);
        SynchroniseObjects(cu_deviceData, m_params);
    }

    __host__ void Host::PerspectiveCamera::ClearRenderState()
    {
        m_hostAccumBuffer->Clear(vec4(0.0f));
        m_hostCompressedRayBuffer->Clear(Cuda::CompressedRay());
        //m_hostPixelFlagsBuffer->Clear(0);
    }

    __global__ void KernelSeedRayBuffer(Device::PerspectiveCamera* camera, const uint frameIdx)
    {
        camera->SeedRayBuffer(kKernelPos<ivec2>(), frameIdx);
    }

    __host__ void Host::PerspectiveCamera::OnPreRenderPass(const float wallTime, const uint frameIdx)
    {
        KernelSeedRayBuffer << < m_gridSize, m_blockSize, 0, m_hostStream >> > (cu_deviceData, frameIdx);
    }

    __global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, const Device::PerspectiveCamera* camera)
    {
        //if (*(deviceOutputImage->AccessSignal()) != kImageWriteLocked) { return; }

        camera->Composite(kKernelPos<ivec2>(), deviceOutputImage);
    }

    __host__ void Host::PerspectiveCamera::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        hostOutputImage->SignalSetWrite(m_hostStream);
        KernelComposite << < m_gridSize, m_blockSize, 0, m_hostStream >> > (hostOutputImage->GetDeviceInstance(), cu_deviceData);
        hostOutputImage->SignalUnsetWrite(m_hostStream);
    }

    __host__ void Host::PerspectiveCamera::GetRawAccumulationData(std::vector<vec4>& rawData, ivec2& dimensions) const
    {
        m_hostAccumBuffer->Download(rawData);
        dimensions = m_hostAccumBuffer->GetMetadata().Dimensions();
    }
}