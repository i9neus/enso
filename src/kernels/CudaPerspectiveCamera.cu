#include "CudaPerspectiveCamera.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"

#define kCameraAA                 1.5f             // The width/height of the anti-aliasing kernel in pixels
#define kCameraSensorSize         0.035f           // The size of the camera sensor in meters
#define kBlades                   5.0f
#define kBladeCurvature           0.0f

namespace Cuda
{    
    // Returns the polar distance r to the perimeter of an n-sided polygon
    __device__ float Ngon(float phi)
    {
        float piBlades = kPi / kBlades;
        float bladeRadius = cos(piBlades) / cos(fmodf(((phi)+piBlades) + piBlades, 2.0f * piBlades) - piBlades);

        // Take into account the blade curvature
        return mix(bladeRadius, 1.0f, kBladeCurvature);
    }

    __device__ Device::PerspectiveCamera::PerspectiveCamera()
    {
        m_useHaltonSpectralSampler = false;
        m_cameraPos = vec2(0.3, 0.5);
        m_cameraLook = vec2(0.5, 0.2);
        m_cameraFLength = vec2(0.45);
        m_cameraFStop = vec2(0.5);
    }
    
    __device__ PackedRay Device::PerspectiveCamera::CreateRay(RenderCtx& renderCtx) const
    {
        // Define our camera vectors and orthonormal basis
        #define kCameraUp vec3(0.0, 1.0, 0.0) 

        // Generate 4 random numbers efrom a continuous uniform distribution
        vec4 xi = renderCtx.pcg();

        // The value of mu is used to sample the spectral wavelength but also the chromatic aberration effect.
        // If we're using the Halton low-disrepancy sampler, hash the input values and sample the sequence
        float mu = xi.y;
        if (m_useHaltonSpectralSampler)
        {
            uint hash = hashCombine(0x01000193u, hashCombine(hashOf(uint(renderCtx.viewportPos.x)), hashOf(uint(renderCtx.viewportPos.y))));
            mu = haltonBase2(hash);
        }

        float theta = kTwoPi * (m_cameraPos.x - 0.5f);
        vec3 cameraPos = vec3(cos(theta), m_cameraPos.y, sin(theta)) * 5.0f * powf(2.0f, mix(-5.0f, 1.0f, m_cameraFLength.x));

        float cameraOriginDist = length(cameraPos);
        vec3 cameraLookAt = vec3(-10.0f * (m_cameraLook - vec2(0.5f)), cameraOriginDist);
        cameraLookAt = createBasis(-cameraPos / cameraOriginDist, kCameraUp) * cameraLookAt;

        vec3 cameraForward = cameraLookAt - cameraPos;
        float focalDistance = length(cameraForward);
        cameraForward /= focalDistance;

        focalDistance *= mix(0.0f, 1.0f, m_cameraFStop.y);
        float fStop = powf(2.0f, mix(-3.0, 8.0, m_cameraFStop.x));

        mat4 basis = createBasis(cameraForward, kCameraUp);

        // Define the focal length and F-number depending, either from built-in or user-defined values
        float focalLength = powf(2.0f, mix(-9.0f, -0.5f, m_cameraFLength.y));

        // Solve the thin-lens equation. http://hyperphysics.phy-astr.gsu.edu/hbase/geoopt/lenseq.html
        float d1 = 0.5 * (focalDistance - sqrt(-4.0 * focalLength * focalDistance + sqr(focalDistance)));
        float d2 = focalDistance - d1; 

        // Generate a position on the sensor, the focal plane, and the lens. This lens will always have circular bokeh
        // but with a few minor additions it's possible to add custom shapes such as irises. We reuse some random numbers
        // but this doesn't really matter because the anti-aliasing kernel is so small that we won't notice the correlation 
        vec2 sensorPos = vec2(xi.z, xi.x) * kCameraSensorSize * kCameraAA / max(renderCtx.viewportSize.x, renderCtx.viewportSize.y) +
            kCameraSensorSize * (renderCtx.viewportPos - vec2(renderCtx.viewportSize) * 0.5) / float(max(renderCtx.viewportSize.x, renderCtx.viewportSize.y));
        vec2 focalPlanePos = vec2(sensorPos.x, sensorPos.y) * d2 / d1;

        vec2 lensPos;
        for (int i = 0; i < 10; i++)
        {
            lensPos = xi.xy * 2.0 - vec2(1.0);
            float bladeDist = Ngon(atan2f(lensPos.y, lensPos.x) + kPi);
            if (length2(lensPos) < sqr(bladeDist)) { break; }
            xi = renderCtx.pcg();
        }
        lensPos *= 0.5 * focalLength / fStop;
        //vec2 lensPos = sampleUnitDisc(xi.xy) * 0.5 * focalLength / fStop;

        // Assemble the ray
        PackedRay ray;
        ray.od.o = basis* vec3(lensPos, d1);
        ray.od.d = xi.xyz;// normalize((basis * vec3(focalPlanePos, focalDistance)) - ray.od.o);
        ray.od.o += cameraPos;
        ray.weight = 1.0f;

        // Sample the wavelength (in Angstroms)  
        ray.lambda = mix(3800.0f, 7000.0f, mu);

        return ray;
    }
}