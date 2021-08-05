#include "../CudaSampler.cuh"
#include "../CudaHash.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRay.cuh"

#include "CudaPerspectiveCamera.cuh"
#include "generic/JsonUtils.h"
#include "../CudaManagedArray.cuh"
#include "../CudaManagedObject.cuh"

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
        viewportDims = ivec2(512, 512);
    }

    __host__ PerspectiveCameraParams::PerspectiveCameraParams(const ::Json::Node& node) :
        PerspectiveCameraParams() 
    { 
        FromJson(node, ::Json::kRequiredWarn);
    }
    
    __host__ void PerspectiveCameraParams::ToJson(::Json::Node& node) const
    {
        node.AddArray("pos", std::vector<float>({ position.x, position.y, position.z }));
        node.AddArray("lookAt", std::vector<float>({ lookAt.x, lookAt.y, lookAt.z }));
        node.AddValue("focalPlane", focalPlane);
        node.AddValue("fLength", fLength);
        node.AddValue("fStop", fStop);
    }

    __host__ void PerspectiveCameraParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetVector("pos", position, flags);
        node.GetVector("lookAt", lookAt, flags);
        node.GetValue("focalPlane", focalPlane, flags);
        node.GetValue("fLength", fLength, flags);
        node.GetValue("fStop", fStop, flags);
    }

    __host__ bool PerspectiveCameraParams::operator==(const PerspectiveCameraParams& rhs) const
    {
        return position == rhs.position &&
            lookAt == rhs.lookAt &&
            focalPlane == rhs.focalPlane &&
            fLength == rhs.fLength &&
            fStop == rhs.fStop;
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

    __device__ void Device::PerspectiveCamera::SeedRayBuffer(const ivec2& viewportPos)
    {
        CompressedRay& compressedRay = (*m_renderState.cu_compressedRayBuffer)[viewportPos.y * 512 + viewportPos.x];

        if (!compressedRay.IsAlive())
        {            
            CreateRay(viewportPos, compressedRay);
        }
    }

    __device__ void Device::PerspectiveCamera::Prepare()
    { 
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
        ray.viewport.x = viewportPos.x;
        ray.viewport.y = viewportPos.y;
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

        // Assemble the ray
        ray.od.o = basis * vec3(lensPos, d1);
        ray.od.d = normalize((basis * vec3(focalPlanePos, focalDistance)) - ray.od.o);
        ray.od.o += cameraPos;
        ray.weight = 1.0f;
        ray.depth = 0;
        ray.flags = kRaySpecular;
        //ray.lambda = mix(3800.0f, 7000.0f, mu);
    }

    __device__ void Device::PerspectiveCamera::Accumulate(RenderCtx& ctx, const vec3& value)
    {
        m_renderState.cu_accumBuffer->Accumulate(ivec2(ctx.emplacedRay.viewport.x, ctx.emplacedRay.viewport.y), value, ctx.depth, ctx.emplacedRay.IsAlive());
    }

    __host__ Host::PerspectiveCamera::PerspectiveCamera(const ::Json::Node& parentNode, const std::string& id) :
        Host::Camera(parentNode, id)
    {
        // Create the accumulation buffer
        m_hostAccumBuffer = AssetHandle<Host::ImageRGBW>(tfm::format("%s_perspAccumBuffer", id), 512, 512, m_hostStream);
        m_hostAccumBuffer->Clear(vec4(0.0f));        
        
        // Instantiate the camera object on the device
        cu_deviceData = InstantiateOnDevice<Device::PerspectiveCamera>();
        FromJson(parentNode, ::Json::kRequiredWarn);

        // Sychronise the device objects
        Device::RenderState renderState;
        renderState.cu_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        renderState.cu_compressedRayBuffer = m_hostCompressedRayBuffer->GetDeviceInstance();
        renderState.cu_blockRayOccupancy = m_hostBlockRayOccupancy->GetDeviceInstance();
        renderState.cu_renderStats = m_hostRenderStats->GetDeviceInstance();
        SynchroniseObjects(cu_deviceData, renderState);

        m_block = dim3(16, 16, 1);
        m_grid = dim3((m_hostAccumBuffer->GetMetadata().Width() + 15) / 16, (m_hostAccumBuffer->GetMetadata().Height() + 15) / 16, 1);
    }

    __host__ AssetHandle<Host::RenderObject> Host::PerspectiveCamera::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kCamera) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::PerspectiveCamera(json, id), id);
    }

    __host__ void Host::PerspectiveCamera::OnDestroyAsset()
    {
        Host::Camera::OnDestroyAsset();
        
        m_hostAccumBuffer.DestroyAsset();
        
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::PerspectiveCamera::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::Camera::FromJson(parentNode, flags);
        
        SynchroniseObjects(cu_deviceData, PerspectiveCameraParams(parentNode));
    }

    __host__ void Host::PerspectiveCamera::ClearRenderState()
    {
        m_hostAccumBuffer->Clear(vec4(0.0f));
        m_hostCompressedRayBuffer->Clear(Cuda::CompressedRay());
        //m_hostPixelFlagsBuffer->Clear(0);
    }

    __global__ void KernelSeedRayBuffer(Device::PerspectiveCamera* camera)
    {
        camera->SeedRayBuffer(kKernelPos<ivec2>());
    }

    __host__ void Host::PerspectiveCamera::SeedRayBuffer()
    {
        KernelSeedRayBuffer << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData);
    }
}