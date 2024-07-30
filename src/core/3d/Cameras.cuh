#pragma once

#include "Ray.cuh"
#include "Basis.cuh"
#include "core/math/Mappings.cuh"

namespace Enso
{
    __host__ __device__ Ray CreatePinholeCameraRay(const vec2& uvView, const vec3& cameraPos, const vec3& cameraLookAt, const float& fov)
    {
        Ray ray;
        ray.od.o = cameraPos;
        ray.od.d = CreateBasis(normalize(cameraPos - cameraLookAt), vec3(0., 1., 0.)) * normalize(vec3(uvView, -tan(toRad(fov))));
        ray.tNear = kFltMax;
        ray.weight = kOne;
        ray.pdf = kFltMax;

        return ray;
    }

    // Returns the polar distance r to the perimeter of an n-sided polygon
    /*__host__ __device__ __forceinline__ float Ngon(float phi)
    {
        const float kBladeCurvature = 0.0f;
        const int kBlades = 5;

        float piBlades = kPi / kBlades;
        float bladeRadius = cos(piBlades) / cos(fmodf(((phi)+piBlades) + piBlades, 2.0f * piBlades) - piBlades);

        // Take into account the blade curvature
        return mix(bladeRadius, 1.0f, kBladeCurvature);
    }

    __host__ __device__ Ray CreateThinLensCameraRay(const vec4& xi, const vec3&, const vec2& uvScreen, const vec3& cameraPos, vec3& cameraLookAt, const vec2& viewRes)
    {
            // Define our camera vectors and orthonormal basis
            constexpr vec3 kCameraUp = vec3(0.0f, 1.0f, 0.0f);

            vec3 cameraForward = cameraLookAt - cameraPos;
            float focalDistance = length(cameraForward);
            cameraForward /= focalDistance;

            const mat3 basis = CreateBasis(cameraForward, kCameraUp);

            // Define the focal length and F-number depending, either from built-in or user-defined values
            constexpr float kCameraFocalLength = 0.050f;
            constexpr float kCameraFStop = 15.0f;
            constexpr float kCameraSensorSize = 0.035f;
            constexpr float kCameraAA = 1.0f;

            // Solve the thin-lens equation. http://hyperphysics.phy-astr.gsu.edu/hbase/geoopt/lenseq.html
            float d1 = 0.5 * (focalDistance - sqrt(-4.0 * kCameraFocalLength * focalDistance + sqr(focalDistance)));
            float d2 = focalDistance - d1;

            // Generate a position on the sensor, the focal plane, and the lens. This lens will always have circular bokeh
            // but with a few minor additions it's possible to add custom shapes such as irises. We reuse some random numbers
            // but this doesn't really matter because the anti-aliasing kernel is so small that we won't notice the correlation 
            vec2 sensorPos = vec2(xi.z, xi.x) * kCameraSensorSize * kCameraAA / fmaxf(viewRes.x, viewRes.y) + uvScreen * kCameraSensorSize;
            vec2 focalPlanePos = vec2(sensorPos.x, sensorPos.y) * d2 / d1;
            vec2 lensPos = SampleUnitDisc(xi.xy) * 0.5 * kCameraFocalLength / kCameraFStop;

            // Assemble the ray
            Ray ray;
            ray.od.o = basis * vec3(lensPos, d1);
            ray.od.d = normalize(mul3(vec3(focalPlanePos, focalDistance), basis) - ray.od.o);
            ray.od.o += cameraPos;
            ray.tNear = kFltMax;
            ray.weight = vec3(1.0, 1.0, 1.0);
            ray.pdf = kFltMax;

            return ray;
        }
    }*/
}