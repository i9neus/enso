#include "CudaSampler.cuh"
#include "CudaHash.cuh"
#include "CudaCtx.cuh"
#include "CudaRay.cuh"
#include "CudaPerspectiveCamera.cuh"
#include "generic/JsonUtils.h"

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
    
    __device__ void Device::PerspectiveCamera::CreateRay(CompressedRay& newRay, RenderCtx& renderCtx) const
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

        // Generate 4 random numbers from a continuous uniform distribution
        vec4 xi = renderCtx.Rand<0, 1, 2, 3>();

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
        vec2 sensorPos = vec2(xi.x, xi.y) * kCameraSensorSize * kCameraAA / max(renderCtx.viewportDims.x, renderCtx.viewportDims.y) +
            kCameraSensorSize * (vec2(renderCtx.viewportPos) - vec2(renderCtx.viewportDims) * 0.5) / float(max(renderCtx.viewportDims.x, renderCtx.viewportDims.y));
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
        newRay.od.o = basis * vec3(lensPos, d1);
        newRay.od.d = normalize((basis * vec3(focalPlanePos, focalDistance)) - newRay.od.o);
        newRay.od.o += cameraPos;
        newRay.weight = 1.0f;
        newRay.depth = 0;
        newRay.flags = 0;
        newRay.lambda = mix(3800.0f, 7000.0f, mu);
        newRay.sampleIdx = renderCtx.sampleIdx;

        newRay.viewport.x = ushort(renderCtx.viewportPos.x);
        newRay.viewport.y = ushort(renderCtx.viewportPos.y);
        newRay.SetAlive();
    }

    __host__ Host::PerspectiveCamera::PerspectiveCamera(const ::Json::Node& parentNode)
    {
        cu_deviceData = InstantiateOnDevice<Device::PerspectiveCamera>();
        FromJson(parentNode, ::Json::kRequiredWarn);
    }

    __host__ AssetHandle<Host::RenderObject> Host::PerspectiveCamera::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kCamera) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::PerspectiveCamera(json), id);
    }

    __host__ void Host::PerspectiveCamera::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::PerspectiveCamera::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::RenderObject::FromJson(parentNode, flags);
        
        SynchroniseObjects(cu_deviceData, PerspectiveCameraParams(parentNode));
    }
}