#include "SplatRasteriser.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/3d/Ctx.cuh"
#include "core/assets/AssetContainer.cuh"
#include "core/3d/Transform.cuh"
#include "core/containers/Vector.cuh"
#include "core/3d/Cameras.cuh"
#include "core/assets/GenericObjectContainer.cuh"
#include "core/math/samplers/MersenneTwister.cuh"
#include "core/math/mat/MatMul.cuh"

#include "../scene/SceneContainer.cuh"
#include "../scene/cameras/Camera.cuh"
#include "../scene/materials/Material.cuh"
#include "../scene/lights/Light.cuh"
#include "../scene/textures/Texture2D.cuh"
#include "../scene/tracables/Tracable.cuh"
#include "../scene/pointclouds/GaussianPointCloud.cuh"

#include "io/json/JsonUtils.h"
//#include "core/AccumulationBuffer.cuh"

//#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
//#include <thrust/device_vector.h>

namespace Enso
{        
    static __device__ bool operator==(const RadixSortKey& lhs, const RadixSortKey& rhs) { return lhs.key == rhs.key; }

    struct RadixSortDecomposer
    {
        __device__::cuda::std::tuple<uint64_t&> operator()(RadixSortKey& ref) const
        {
            return { ref.key };
        }
    };

    __device__ void SplatRasteriserObjects::Validate() const
    {
        CudaAssert(frameBuffer);
        CudaAssert(numSplatRefs);

        if (splatList && projectedSplats)
        {
            CudaAssert(splatList->size() == projectedSplats->size());
        }
      
        if (tileRanges) { CudaAssert(tileRanges->size() > 0); }
    }

    __host__ __device__ void Device::SplatRasteriser::Synchronise(const SplatRasteriserParams& params)
    {
        m_params = params;
    }

    __device__ void Device::SplatRasteriser::Synchronise(const SplatRasteriserObjects& objects)
    {
        m_objects = objects;
    }

    // Safe way of extracting the bits from a float without violating strict aliasing rules
    __device__ __forceinline__ uint32_t FloatBits(const float& f)
    {
        union { float f; uint32_t i; } u = { f };
        return u.i;
    }

    __device__ void Device::SplatRasteriser::Verify()
    {
        if (kKernelIdx == 0)
        {
            for (int idx = 0; idx < m_objects.projectedSplats->size(); ++idx)
            {
                const uint32_t b = (*m_objects.projectedSplats)[idx].bounds;
                
                //uint32_t hi = uint32_t((*m_objects.sortedKeys)[idx].key >> 32);
                //uint32_t lo = uint32_t((*m_objects.sortedKeys)[idx].key & 0xffffffff);
                printf("%i: %i, %i, %i, %i\n", idx, (b & 255), (b >> 8) & 255, (b >> 16) & 255, (b >> 24) & 255);
            }          
        }
    }
    DEFINE_KERNEL_PASSTHROUGH(Verify);

    __device__ void Device::SplatRasteriser::ProjectSplats()
    {
        if (kKernelIdx >= m_objects.splatList->size()) { return; }      

        // Load some data into shared memory and sync the block
        __shared__ mat3 W;
        __shared__ vec3 camPos;
        __shared__ float camFov;
        __shared__ float screenRatio;
        __shared__ uvec2 gridDims;
        if (kThreadIdx == 0)
        {
            // Sanity check
            CudaAssertDebugMsg(m_objects.splatList->size() == m_objects.projectedSplats->size(), "Splat lists size mismatch");
            CudaAssertDebugMsg(m_objects.splatList->size() == m_objects.splatPMF->size(), "PMF size mismatch");
            
            const auto& params = m_objects.activeCamera->GetCameraParams();
            W = params.inv;
            camPos = params.cameraPos;
            camFov = params.cameraFov;
            screenRatio = float(m_params.viewport.dims.y) / float(m_params.viewport.dims.x);
            gridDims = m_params.tileGrid.dims;
        }
        __syncthreads();

        const auto& splatWorld = (*m_objects.splatList)[kKernelIdx];
        auto& splatCamera = (*m_objects.projectedSplats)[kKernelIdx];

        // Project the position of the splat into camera space
        const vec3 pCam = W * (splatWorld.p - camPos);

        // Create rotation and transpose product of scale matrices
        const mat3 R = splatWorld.rot.RotationMatrix();
        const mat3 ST(vec3(splatWorld.sca.x * splatWorld.sca.x, 0.0f, 0.0f),
                      vec3(0., splatWorld.sca.y * splatWorld.sca.y, 0.0),
                      vec3(0., 0.0, splatWorld.sca.z * splatWorld.sca.z));

        // Jacobian of projective approximation (Zwicker et al)
        const float lenPCam = length(pCam);
        const mat3 J = mat3(vec3(1. / pCam.z, 0.0, pCam.x / lenPCam),
                       vec3(0., 1. / pCam.z, pCam.y / lenPCam),
                       vec3(-pCam.x / (pCam.z * pCam.z), -pCam.y / (pCam.z * pCam.z), pCam.z / lenPCam));

        // Homogenise the projected camera position
        vec3 pView = vec3(pCam.xy / (pCam.z * -tanf(toRad(camFov))), -pCam.z);

        // Initialise the splat hashes
        if (pView.z <= 0.f || pView.x < -1.f || pView.x > 1.f || pView.y < -screenRatio || pView.y > screenRatio)
        {
            // Objects outside of the view frustum are set to zero
            splatCamera.key = 0;
            splatCamera.tileIdx = 0;
            (*m_objects.splatPMF)[kKernelIdx] = 0;
        }
        else
        {
            // Build covariance matrix
            const mat3 cov = R * ST * transpose(R);

            // Project covariance matrix
            const mat3 sigma3 = J * W * cov * transpose(W) * transpose(J);
            const mat2 sigma2(sigma3[0].xy, sigma3[1].xy);

            constexpr float kSplatEigenBound = 3.0f;
            const float tr = 0.5 * trace(sigma2);
            const float bound = (sqrtf((tr + sqrtf(tr * tr - det(sigma2)))) * kSplatEigenBound) * m_params.viewport.dims.x;

            // Compose the projected splat
            splatCamera.sigma = inverse(sigma2);
            splatCamera.p = pView;
            splatCamera.rgba = splatWorld.rgba;

            // Populate the key-value pairs ready for sorting
            const vec2 pScreen = NormalisedScreenToPixel(pView.xy, m_params.viewport.dims);
            const ivec2 ijTile = ivec2(pScreen) / int(kSplatRasteriserTileSize);

            constexpr int32_t kMaxTileBounds = 1;
            const int32_t uPlus = min3(kMaxTileBounds, m_params.tileGrid.dims.x - ijTile.x, int32_t(pScreen.x + bound) / kSplatRasteriserTileSize - ijTile.x);
            const int32_t uMinus = min3(kMaxTileBounds, ijTile.x, ijTile.x - int32_t(pScreen.x - bound) / kSplatRasteriserTileSize);
            const int32_t vPlus = min3(kMaxTileBounds, m_params.tileGrid.dims.y - ijTile.y, int32_t(pScreen.y + bound) / kSplatRasteriserTileSize - ijTile.y);
            const int32_t vMinus = min3(kMaxTileBounds, ijTile.y, ijTile.y - int32_t(pScreen.y - bound) / kSplatRasteriserTileSize);
            //const int32_t uPlus = 0, uMinus = 0, vPlus = 0, vMinus = 0;

            splatCamera.bounds = vMinus | (vPlus << 8) | (uMinus << 16) | (uPlus << 24);

            const uint32_t tileIdx = ijTile.y * m_params.tileGrid.dims.x + ijTile.x;          
            splatCamera.tileIdx = tileIdx;

            //printf("%i: %i: %f, %f, %f\n", kKernelIdx, tileIdx + 1, pView.x, pView.y, pView.z);
            //printf("%i: %i, %i, %i, %i\n", kKernelIdx, uPlus, uMinus, vPlus, vMinus);

            // Set the key as the concatenation of the grid index and the splat depth
            // We add 1 so that culled splats are all sorted to bucket zero
            splatCamera.key = (uint64_t(1 + tileIdx) << 32) | uint64_t(FloatBits(pView.z));

            // Set the tile coverage of this splat ready so we can duplicate it before sorting
            (*m_objects.splatPMF)[kKernelIdx] = (1 + uPlus + uMinus) * (1 + vPlus + vMinus);
        }
    }
    DEFINE_KERNEL_PASSTHROUGH(ProjectSplats);

    __device__ void Device::SplatRasteriser::Debug(const uint32_t op)
    {
        if (op == 0)
        {
            for (int i = 0; i < m_objects.splatPMF->size(); ++i)
            {
                printf("%i: %i\n", i, (*m_objects.splatPMF)[i]);
            }
        }
        else if(op == 1)
        {
            for (int i = 0; i < m_objects.splatCMF->size(); ++i)
            {
                printf("%i: %i\n", i, (*m_objects.splatCMF)[i]);
            }
        }
        else if (op == 2)
        {
            uint32_t actualMaxTile = 0;
            for (int i = 0; i < m_objects.projectedSplats->size(); ++i)
            {
                auto idx = (*m_objects.projectedSplats)[i].tileIdx;
                //auto key = uint32_t((*m_objects.unsortedKeys)[i].key >> 32);
                actualMaxTile = max(actualMaxTile, idx);
                //printf("  - %i: %i, %i\n", i, idx, key);
            }
            /*for (int i = 0; i < m_objects.splatCMF->size(); ++i)
            {
                printf("  - %i: %i\n", i, (*m_objects.splatCMF)[i]);
            }*/
            const uint32_t sortedMaxTile = uint32_t((*m_objects.sortedKeys)[m_objects.sortedKeys->size() - 1].key >> 32);
            printf("%i -> %i:\n", sortedMaxTile, actualMaxTile);
        }       
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Debug);

    // Finds the highest set bit for a value of a given integral type
    template<typename T>
    __device__ __forceinline__ int8_t HighestBitSet(const T& v)
    {
        int8_t a = 0, b = sizeof(T) * 8 - 1;
        while (b - a > 1)
        {
            const uint8_t h = a + ((b - a) >> 1);
            if (v >> h) { a = h; }
            else { b = h; }
        }
        return (v >> b) ? a : b;
    }
    
    __device__ void Device::SplatRasteriser::MapCMF(uint32_t level)
    {
        const uint32_t span = 1 << level;
        const uint32_t idx = uint32_t(kKernelIdx) << level;
        if (idx + span - 1 < m_objects.splatPMF->size())
        {
            (*m_objects.splatPMF)[idx + span - 1] += (*m_objects.splatPMF)[idx + (span >> 1) - 1];
        }      
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(MapCMF);

    __device__ void Device::SplatRasteriser::ReduceCMF()
    {
        if (kKernelIdx < m_objects.splatCMF->size())
        {
            const int8_t highBit = HighestBitSet<uint32_t>(kKernelIdx + 1) - 1;
            uint32_t sum = 0, idx = 0;
            for (int8_t bit = highBit; bit >= 0; --bit)
            {
                if (((kKernelIdx + 1) >> bit) & 1)
                {
                    idx += 1 << bit;
                    sum += (*m_objects.splatPMF)[idx - 1];
                }
            }
            (*m_objects.splatCMF)[kKernelIdx] = sum;

            // Make a note of the cumulative number of overlapped tiles
            if (kKernelIdx == m_objects.splatCMF->size() - 1) { *m_objects.numSplatRefs = sum; }
        }        
    }
    DEFINE_KERNEL_PASSTHROUGH(ReduceCMF);

    __device__ void Device::SplatRasteriser::PopulateKeyRefPairs()
    {
        if (kKernelIdx >= m_objects.splatCMF->size()) { return; }

        const uint32_t start = (kKernelIdx == 0) ? 0 : (*m_objects.splatCMF)[kKernelIdx - 1];
        const uint32_t end = (*m_objects.splatCMF)[kKernelIdx];
        if (start != end)
        {
            const ProjectedGaussianPoint& splat = (*m_objects.projectedSplats)[kKernelIdx];
            const uint64_t depth = splat.key & 0xffffffffull;
            const int32_t tileIdx = (splat.key >> 32) - 1;
            int32_t refIdx = start;
            CudaAssertDebug(tileIdx >= 0);

            const int32_t uTile = tileIdx % m_params.tileGrid.dims.x, vTile = tileIdx / m_params.tileGrid.dims.x;

            for (int v = vTile - (splat.bounds & 0xff); v <= vTile + ((splat.bounds >> 8) & 0xff); ++v)
            {
                for (int u = uTile - ((splat.bounds >> 16) & 0xff); u <= uTile + ((splat.bounds >> 24) & 0xff); ++u, ++refIdx)
                {
                    // This should never happen
                    if (refIdx == end)
                    {
                        const uint32_t splatArea = (1 + (splat.bounds & 0xff) + ((splat.bounds >> 8) & 0xff)) * (1 + ((splat.bounds >> 16) & 0xff) + ((splat.bounds >> 24) & 0xff));
                        CudaAssertFmt(refIdx < end, "Out of bounds! %i -> %i", splatArea, end - start);
                    }

                    (*m_objects.unsortedRefs)[refIdx] = kKernelIdx;
                    (*m_objects.unsortedKeys)[refIdx] = (uint64_t(v * m_params.tileGrid.dims.x + u) << 32) | depth;
                }
            }
        }
    }
    DEFINE_KERNEL_PASSTHROUGH(PopulateKeyRefPairs);

    __device__ void Device::SplatRasteriser::ClearTileRanges()
    {
        if (kKernelIdx < m_objects.tileRanges->size())
        {
            (*m_objects.tileRanges)[kKernelIdx] = uvec2(0xffffffffu, 0x0u);
        }
    }
    DEFINE_KERNEL_PASSTHROUGH(ClearTileRanges);

    __device__ void Device::SplatRasteriser::DetermineTileRanges()
    {
        if (kKernelIdx >= *m_objects.numSplatRefs) { return; }

        CudaAssertDebugMsg(*m_objects.numSplatRefs <= m_objects.sortedKeys->size(), "Num refs exceeds key array size");
        CudaAssertDebugFmt(m_objects.tileRanges->size() == m_params.tileGrid.numTiles + 1, "Tile grid memory error (%i, %i)", m_objects.tileRanges->size(), m_params.tileGrid.numTiles + 1);

        if (kKernelIdx == 0)
        {
            (*m_objects.tileRanges)[(*m_objects.sortedKeys)[0].key >> 32][0] = 0;
        }
        else if (kKernelIdx == m_objects.sortedRefs->size() - 1)
        {
            (*m_objects.tileRanges)[(*m_objects.sortedKeys)[kKernelIdx].key >> 32][1] = kKernelIdx;
        }
        else
        {
            const uint32_t prevTile = (*m_objects.sortedKeys)[kKernelIdx - 1].key >> 32;
            const uint32_t thisTile = (*m_objects.sortedKeys)[kKernelIdx].key >> 32;
            const uint32_t nextTile = (*m_objects.sortedKeys)[kKernelIdx + 1].key >> 32;
            if (prevTile != thisTile)
            {
                (*m_objects.tileRanges)[thisTile][0] = kKernelIdx;
            }
            if (nextTile != thisTile)
            {
                (*m_objects.tileRanges)[thisTile][1] = kKernelIdx;
            }
        }
    }
    DEFINE_KERNEL_PASSTHROUGH(DetermineTileRanges);

    __device__ void Device::SplatRasteriser::RenderSortedSplats()
    {
        const ivec2 xyViewport = kKernelPos<ivec2>();
        if (xyViewport.x < 0 || xyViewport.x >= m_params.viewport.dims.x || xyViewport.y < 0 || xyViewport.y >= m_params.viewport.dims.y) { return; }

        //__syncthreads();

        CudaAssertDebug(m_params.tileGrid.numTiles + 1 == m_objects.tileRanges->size()); // Sanity check      
        const uint32_t tileIdx = (xyViewport.y / kSplatRasteriserTileSize) * m_params.tileGrid.dims.x + (xyViewport.x / kSplatRasteriserTileSize);
        CudaAssertDebug(tileIdx < m_objects.tileRanges->size());

        vec3 L = kZero;
        const uvec2 range = (*m_objects.tileRanges)[tileIdx];
        CudaAssertFmt(range.y < m_objects.sortedRefs->size(), "range.y < m_objects.sortedRefs->size(): %i -> %i", range.y, m_objects.sortedRefs->size());
        if (range[0] <= range[1])
        {
            const vec2 uvView = PixelToNormalisedScreen(vec2(xyViewport), vec2(m_params.viewport.dims));
            float alpha = 1;

            for (uint32_t refIdx = range[0]; refIdx <= range[1] && alpha > 1e-3f; ++refIdx)
            {                
                CudaAssertDebugFmt(refIdx < m_objects.sortedRefs->size(), "refIdx < m_objects.sortedRefs->size(): %i -> %i", refIdx, m_objects.sortedRefs->size());
                const int splatIdx = (*m_objects.sortedRefs)[refIdx];

                if (splatIdx > m_objects.projectedSplats->size()) { continue; }

                CudaAssertDebugFmt(splatIdx < m_objects.projectedSplats->size(), "splatIdx < m_objects.projectedSplats->size(): %i (%i-%i): %i -> %i", refIdx, range[0], range[1], splatIdx, m_objects.projectedSplats->size());
                const auto& splat = (*m_objects.projectedSplats)[splatIdx];

                // Gaussian PDF
                const vec2 mu = uvView - splat.p.xy;
                const float G = expf(-0.5 * dot(mu * splat.sigma, mu)) * splat.rgba.w;
                //const float G = (length(mu) < 0.001f) ? 1.0 : 0.0f;    

                // Splat 
                //L = mix(L, splat.rgba.xyz, G * splat.rgba.w);                
                //L += kGreen * 0.5;
                L += splat.rgba.xyz * G  * alpha;

                // Attenuate alpha 
                alpha *= 1 - G;
            }
        }

        m_objects.frameBuffer->At(xyViewport) = vec4(L, 1.0f);

    }
    DEFINE_KERNEL_PASSTHROUGH(RenderSortedSplats);

    __device__ void Device::SplatRasteriser::RenderSplatTiles()
    {
        const ivec2 xyViewport = kKernelPos<ivec2>();
        if (xyViewport.x < 0 || xyViewport.x >= m_params.viewport.dims.x || xyViewport.y < 0 || xyViewport.y >= m_params.viewport.dims.y) { return; }

        CudaAssertDebug(m_params.tileGrid.numTiles + 1 == m_objects.tileRanges->size()); // Sanity check      

        const vec2 uvView = PixelToNormalisedScreen(vec2(xyViewport), vec2(m_params.viewport.dims));
        const uint32_t tileIdx = 1 + (xyViewport.y / kSplatRasteriserTileSize) * m_params.tileGrid.dims.x + (xyViewport.x / kSplatRasteriserTileSize);
        CudaAssertDebug(tileIdx < m_objects.tileRanges->size());

        const uvec2 range = (*m_objects.tileRanges)[tileIdx];

        if (range[0] <= range[1])
        {
            auto& L = m_objects.frameBuffer->At(xyViewport);
            L = Blend(L, Hue(HashOfAsFloat(float(range[1] - range[0]) / (float(m_params.tileGrid.numTiles) * float(m_objects.sortedRefs->size())))), 0.5f);
        }

        /*for (int idx = 0; idx < m_objects.unsortedKeys->size(); ++idx)
        {
            if (((*m_objects.unsortedKeys)[idx].key >> 32) == tileIdx)
            {
                auto& L = m_objects.frameBuffer->At(xyViewport);
                L = Blend(L, Hue(HashOfAsFloat(tileIdx)), 0.5f);
                return;
            }
        }*/
    }
    DEFINE_KERNEL_PASSTHROUGH(RenderSplatTiles);

    // Renders the splats directly without sorting
    __device__ void Device::SplatRasteriser::RenderUnsortedSplats()
    {
        const ivec2 xyViewport = kKernelPos<ivec2>();
        if (xyViewport.x < 0 || xyViewport.x >= m_params.viewport.dims.x || xyViewport.y < 0 || xyViewport.y >= m_params.viewport.dims.y) { return; }

        // Load some data into shared memory and sync the block
        __shared__ mat3 W;
        __shared__ vec3 camPos;
        __shared__ float camFov;
        __shared__ float screenRatio;
        if (kThreadIdx == 0)
        {
            const auto& params = m_objects.activeCamera->GetCameraParams();
            W = params.inv;
            camPos = params.cameraPos;
            camFov = params.cameraFov;
            screenRatio = float(m_params.viewport.dims.y) / float(m_params.viewport.dims.x);
        }
        __syncthreads();

        const vec2 uvView = PixelToNormalisedScreen(vec2(xyViewport), vec2(m_params.viewport.dims));
        vec3 L = kZero;
        
        ////////////////
        //const uint32_t tileIdx = (xyViewport.y / kSplatRasteriserTileSize) * m_params.tileGrid.dims.x + (xyViewport.x / kSplatRasteriserTileSize);
        ////////////////

        for (int idx = 0; idx < m_objects.splatList->size(); ++idx)
        {
            const auto& splat = (*m_objects.splatList)[idx];

            // Project the position of the splat into camera space
            const vec3 pCam = W * (splat.p - camPos);
            const vec3 pView = vec3(pCam.xy / (pCam.z * -tanf(toRad(camFov))), -pCam.z);

            if (pView.z > 0.f && pView.x > -1.f && pView.x < 1.f && pView.y > -screenRatio && pView.y < screenRatio)
            {
                // Create rotation and transpose product of scale matrices
                const mat3 R = splat.rot.RotationMatrix();
                const mat3 ST(vec3(splat.sca.x * splat.sca.x, 0.0f, 0.0f),
                    vec3(0., splat.sca.y * splat.sca.y, 0.0),
                    vec3(0., 0.0, splat.sca.z * splat.sca.z));

                // Jacobian of projective approximation (Zwicker et al)
                const float lenPCam = length(pCam);
                const mat3 J = mat3(vec3(1. / pCam.z, 0.0, pCam.x / lenPCam),
                    vec3(0., 1. / pCam.z, pCam.y / lenPCam),
                    vec3(-pCam.x / (pCam.z * pCam.z), -pCam.y / (pCam.z * pCam.z), pCam.z / lenPCam));

                // Build covariance matrix
                const mat3 cov = R * ST * transpose(R);

                // Project covariance matrix
                const mat3 sigma3 = J * W * cov * transpose(W) * transpose(J);
                const mat2 sigma2 = mat2(sigma3[0].xy, sigma3[1].xy);

                // Gaussian PDF
                const vec2 mu = uvView - pView.xy;
                const float G = expf(-0.5 * dot(mu * inverse(sigma2), mu));

                // Splat 
                L = mix(L, splat.rgba.xyz, G * splat.rgba.w);

                ////////////////
                /*const vec2 pScreen = NormalisedScreenToPixel(pView.xy, m_params.viewport.dims);
                const ivec2 ijTile = ivec2(pScreen) / int(kSplatRasteriserTileSize);
                const uint32_t splatTileIdx = ijTile.y * m_params.tileGrid.dims.x + ijTile.x;
                if (tileIdx == splatTileIdx) L += kRed * 0.5;*/
                ////////////////
            }
        }

        vec4& pixel = m_objects.frameBuffer->At(xyViewport);
        //pixel = Blend(pixel, L, 0.5f);
        pixel.xyz += L;
    }
    DEFINE_KERNEL_PASSTHROUGH(RenderUnsortedSplats);

    __host__ __device__ vec4 Device::SplatRasteriser::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        if (!GetWorldBBox().Contains(pWorld)) { return vec4(0.0f); }

#ifdef __CUDA_ARCH__
        const vec2 pObject = ToObjectSpace(pWorld);
        const ivec2 pPixel = ivec2(vec2(m_params.viewport.dims) * (pObject - m_params.viewport.objectBounds.lower) / m_params.viewport.objectBounds.Dimensions());

        if (!m_params.hasValidSplatCloud)
        {
            const float hatch = step(0.5f, fract(0.05f * dot(pObject / viewCtx.dPdXY, vec2(1.f))));
            return vec4(kOne * hatch * 0.1f, 1.f);
        }
        else if (pPixel.x >= 0 && pPixel.x < m_params.viewport.dims.x && pPixel.y >= 0 && pPixel.y < m_params.viewport.dims.y)
        {
            return m_objects.frameBuffer->At(pPixel);
        }
#else
        return vec4(1.);
#endif
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __host__ AssetHandle<Host::GenericObject> Host::SplatRasteriser::Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects)
    {
        return AssetAllocator::CreateChildAsset<Host::SplatRasteriser>(parentAsset, id, genericObjects);
    }

    __host__ Host::SplatRasteriser::SplatRasteriser(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects) :
        DrawableObject(initCtx, &m_hostInstance),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::SplatRasteriser>(*this)),
        m_hostNumSplatRefs(0)
    {        
        DrawableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::DrawableObject>(cu_deviceInstance));
        RenderableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::RenderableObject>(cu_deviceInstance));

        constexpr int kViewportWidth = 1200;
        constexpr int kViewportHeight = 675;

        const vec2 boundHalf = 0.25 * ((kViewportHeight > kViewportWidth) ?
            vec2(1.f, float(kViewportHeight) / float(kViewportWidth)) :
            vec2(float(kViewportWidth) / float(kViewportHeight), 1.f));

        m_params.viewport.dims = uvec2(kViewportWidth, kViewportHeight);
        m_params.viewport.objectBounds = BBox2f(-boundHalf, boundHalf);
        m_params.tileGrid.dims = uvec2(ceil(vec2(m_params.viewport.dims) / float(kSplatRasteriserTileSize)));
        m_params.tileGrid.numTiles = Area(m_params.viewport.dims);
        m_wallTime.Reset();

        // Create some Cuda objects
        m_hostFrameBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "framebuffer", kViewportWidth, kViewportHeight, nullptr);
        m_hostProjectedSplats = AssetAllocator::CreateChildAsset<Host::Vector<ProjectedGaussianPoint>>(*this, "projectedpointlist", 0u, kVectorDeviceOnly);
        m_hostUnsortedKeys = AssetAllocator::CreateChildAsset<Host::Vector<RadixSortKey>>(*this, "unsortedkeys", 0u, kVectorDeviceOnly | kVectorNoShrinkDeviceCapacity);
        m_hostSortedKeys = AssetAllocator::CreateChildAsset<Host::Vector<RadixSortKey>>(*this, "sortedkeys", 0u, kVectorDeviceOnly | kVectorNoShrinkDeviceCapacity);
        m_hostUnsortedRefs = AssetAllocator::CreateChildAsset<Host::Vector<uint32_t>>(*this, "unsortedrefs", 0u, kVectorDeviceOnly | kVectorNoShrinkDeviceCapacity);
        m_hostSortedRefs = AssetAllocator::CreateChildAsset<Host::Vector<uint32_t>>(*this, "sortedrefs", 0u, kVectorDeviceOnly | kVectorNoShrinkDeviceCapacity);
        m_radixSortTempStorage = AssetAllocator::CreateChildAsset<Host::Vector<uint8_t>>(*this, "radixsorttemp", 0u, kVectorDeviceOnly);
        m_hostTileRanges = AssetAllocator::CreateChildAsset<Host::Vector<uvec2>>(*this, "tileranges", 1 + m_params.tileGrid.numTiles, kVectorDeviceOnly);
        m_hostSplatPMF = AssetAllocator::CreateChildAsset<Host::Vector<uint32_t>>(*this, "maptileoverlapcdf", 0u, kVectorDeviceOnly);
        m_hostSplatCMF = AssetAllocator::CreateChildAsset<Host::Vector<uint32_t>>(*this, "rerducetileoverlapcdf", 0u, kVectorDeviceOnly);

        m_objects.frameBuffer = m_hostFrameBuffer->GetDeviceInstance();
        m_objects.projectedSplats = m_hostProjectedSplats->GetDeviceInstance();
        m_objects.unsortedKeys = m_hostUnsortedKeys->GetDeviceInstance();
        m_objects.sortedKeys = m_hostSortedKeys->GetDeviceInstance();
        m_objects.unsortedRefs = m_hostUnsortedRefs->GetDeviceInstance();
        m_objects.sortedRefs = m_hostSortedRefs->GetDeviceInstance();
        m_objects.tileRanges = m_hostTileRanges->GetDeviceInstance();
        m_objects.splatPMF = m_hostSplatPMF->GetDeviceInstance();
        m_objects.splatCMF = m_hostSplatCMF->GetDeviceInstance();
        m_objects.numSplatRefs = m_unifiedNumSplatRefs.GetDevicePtr();

        Synchronise(kSyncObjects | kSyncParams);

        Cascade({ kDirtySceneObjectChanged });

        IsOk(cudaDeviceSynchronize());
    }

    __host__ Host::SplatRasteriser::~SplatRasteriser() noexcept
    {
        m_hostFrameBuffer.DestroyAsset();
        m_hostProjectedSplats.DestroyAsset();
        m_hostUnsortedKeys.DestroyAsset();
        m_hostSortedKeys.DestroyAsset();
        m_hostUnsortedRefs.DestroyAsset();
        m_hostSortedRefs.DestroyAsset();
        m_radixSortTempStorage.DestroyAsset();
        m_hostTileRanges.DestroyAsset();
        m_hostSplatPMF.DestroyAsset();
        m_hostSplatCMF.DestroyAsset();

        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::SplatRasteriser::OnSynchroniseDrawableObject(const uint syncFlags)
    {
        // Only sync the objects if a SceneContainer has been bound
        if (syncFlags & kSyncObjects)
        {
            SynchroniseObjects<Device::SplatRasteriser>(cu_deviceInstance, m_objects);
        }
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::SplatRasteriser>(cu_deviceInstance, m_params);
            m_hostInstance.Synchronise(m_params);
        }
    }

    __host__ void Host::SplatRasteriser::RebuildSplatCloud()
    {
        if (!m_hostSceneContainer || !m_gaussianPointCloud) { return; }

        Log::Debug("Building splat cloud...");

        auto& tracables = m_hostSceneContainer->Tracables();
        constexpr int kTotalSplats = 50000;
        constexpr float kSplatAreaGain = 1.f;
        float totalSurfaceArea = 0.f;
        int numGeneratedSplats = 0;

        std::vector<float> tracableAreas(tracables.size());
        MersenneTwister rng(987659672);

        Log::Debug("Geometry surface area:");
        for (int idx = 0; idx < tracables.size(); ++idx)
        {
            tracableAreas[idx] = tracables[idx]->CalculateSurfaceArea();
            totalSurfaceArea += tracableAreas[idx];
            Log::Debug("  - %s: %f", tracables[idx]->GetAssetID(), tracableAreas[idx]);
        }

        for (int idx = 0; idx < m_hostSceneContainer->Tracables().size(); ++idx)
        {
            const int numSplats = std::max(1, int(std::ceil(kTotalSplats * tracableAreas[idx] / totalSurfaceArea)));
            auto splatList = tracables[idx]->GenerateGaussianPointCloud(numSplats, kSplatAreaGain, rng);
            m_gaussianPointCloud->AppendSplats(splatList);
            numGeneratedSplats += splatList.size();
        }

        Log::Debug("Created cloud containing %i splats", numGeneratedSplats);

        m_gaussianPointCloud->Finalise();

        // Allocate data for the projections
        const size_t numSplats = m_gaussianPointCloud->GetSplatList().size();
        m_hostProjectedSplats->resize(numSplats);
        m_hostSplatPMF->resize(numSplats);
        m_hostSplatCMF->resize(numSplats);

        // Pre-allocate sorting data
        //SortSplats(true);
    }

    __host__ void Host::SplatRasteriser::Bind(GenericObjectContainer& objects)
    {
        // REMOVE THIS!!!
        if (m_gaussianPointCloud) return;

        m_hostSceneContainer = objects.FindFirstOfType<Host::SceneContainer>();
        if (!m_hostSceneContainer)
        {
            Log::Warning("Warning! Splat rasteriser '%s' could not bind to a valid SceneContainer object.", GetAssetID());
        }
        else
        {
            if (m_hostSceneContainer->Cameras().empty())
            {
                Log::Warning("Warning! Splat rasteriser '%s' found no cameras in the scene.");
                m_hostActiveCamera = nullptr;
            }
            else
            {
                m_hostActiveCamera = m_hostSceneContainer->Cameras().back();
                m_objects.activeCamera = m_hostActiveCamera->GetDeviceInstance();
            }
        }

        m_gaussianPointCloud = objects.FindFirstOfType<Host::GaussianPointCloud>();
        if (m_gaussianPointCloud)
        {
            m_objects.splatList = m_gaussianPointCloud->GetSplatList().GetDeviceInstance();
            m_params.hasValidSplatCloud = true;

            RebuildSplatCloud();
        }
        else
        {
            Log::Warning("Warning! Splat rasteriser '%s' could not bind to a valid GaussianPointCloud object.", GetAssetID());
            m_objects.splatList = nullptr;
            m_objects.projectedSplats = nullptr;
            m_params.hasValidSplatCloud = false;
        }

        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ bool Host::SplatRasteriser::ComputeTileOverlaps()
    {
        // Splats often overlap their tiles, so we need to duplicate references to them across any tiles they overlap.
        // The projection step determines the extent to which a splat spills over on adjacent tiles. Here, we define the cumulative mass 
        // per splat that can then be used to index the reference list.

        //KernelDebug << <1, 1 >> > (cu_deviceInstance, 0u);
        //IsOk(cudaDeviceSynchronize());

        // First, build an in-place summed value table by mapping successive iterations of the mass. Complexity O(n log n) for n splats.
        // If the array looks like { 1, 1, 1, 1, 1, 1, 1, 1 } then its map looks like { 1, 2, 1, 4, 1, 2, 1, 8 }
        for (int range = m_hostSplatPMF->size() >> 1, interval = 0;
            range >= 1;
            range >>= 1, ++interval)
        {
            //Log::Debug("Range: %i, interval: %i", range, interval);
            const int numBlocks = (range + 255) >> 8;
            KernelMapCMF << <numBlocks, 256 >> > (cu_deviceInstance, interval + 1);
        }

        // Reduce the value table data to build the final CMF.  The reduced array is a running sum of the values in the ref table. 
        // For the example about, this looks like { 1, 2, 3, 4, 5, 6, 7, 8 }. Complexity O(n log n)
        const int numSplatBlocks = (m_hostSplatPMF->size() + 255) >> 8;
        KernelReduceCMF << < numSplatBlocks, 256 >> > (cu_deviceInstance);

        //KernelDebug << <1, 1 >> > (cu_deviceInstance, 1u);
        IsOk(cudaDeviceSynchronize());

        // The total number of references is stored in the last entry of m_hostSplatPMF
        m_hostNumSplatRefs = m_unifiedNumSplatRefs;
        Log::Warning("Visible splats: %i", m_hostNumSplatRefs);

        // Bail out if no splats are visible
        if (m_hostNumSplatRefs == 0) { return false; }

        // Limit to ten million referenced splats
        constexpr uint32_t kMaxSplatRefs = 1e8;
        AssertMsgFmt(m_hostNumSplatRefs <= kMaxSplatRefs, "Total number of refs %i exceeds hard upper limit %i", m_hostNumSplatRefs, kMaxSplatRefs);

        // If necessary, grow the key/ref arrays to accommodate the number of duplicated splats
        const auto delta = m_hostUnsortedKeys->grow(m_hostNumSplatRefs);
        m_hostSortedKeys->grow(m_hostNumSplatRefs);
        m_hostUnsortedRefs->grow(m_hostNumSplatRefs);
        m_hostSortedRefs->grow(m_hostNumSplatRefs);

        if (delta > 0)
        {
            Log::Debug("Grew key/ref arrays by %i elements", delta);
        }

        // Populate the key/ref pairs by 
        KernelPopulateKeyRefPairs <<< numSplatBlocks, 256>>>(cu_deviceInstance);

        IsOk(cudaDeviceSynchronize());     

        return true;
    }

    __host__ void Host::SplatRasteriser::SortSplats()
    {
        Assert(m_hostUnsortedKeys->size() == m_hostSortedKeys->size());
        Assert(m_hostUnsortedRefs->size() == m_hostSortedRefs->size());
        Assert(m_hostUnsortedRefs->size() == m_hostUnsortedKeys->size());
        Assert(m_hostNumSplatRefs > 0);

        const RadixSortKey* keysInPtr = m_hostUnsortedKeys->GetDeviceData();
        RadixSortKey* keyOutPtr = m_hostSortedKeys->GetDeviceData();
        const uint32_t* valInPtr = m_hostUnsortedRefs->GetDeviceData();
        uint32_t* valsOutPtr = m_hostSortedRefs->GetDeviceData();

        HighResolutionTimer timer;        
        size_t tempStorageBytes = 0u;

        cub::DeviceRadixSort::SortPairs(
                nullptr,
                tempStorageBytes,
                keysInPtr,
                keyOutPtr,
                valInPtr,
                valsOutPtr,
                m_hostNumSplatRefs,
                RadixSortDecomposer{},
                0,
                64);   

        if (m_radixSortTempStorage->size() != tempStorageBytes)
        {
            m_radixSortTempStorage->resize(tempStorageBytes);
            Log::Debug("Resized temp storage to %i bytes.", tempStorageBytes);
        }

        uint8_t* tempStorage = m_radixSortTempStorage->GetDeviceData();

        cub::DeviceRadixSort::SortPairs(
            tempStorage,
            tempStorageBytes,
            keysInPtr,
            keyOutPtr,
            valInPtr,
            valsOutPtr,
            m_hostNumSplatRefs,
            RadixSortDecomposer{},
            0,
            64);

        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::SplatRasteriser::Render()
    {        
        if (!m_hostSceneContainer || !m_hostActiveCamera) { return; }
        if (m_hostProjectedSplats->size() == 0) { return; }
                
        //KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        //if (m_params.frameIdx > 10) return;

        if (RenderableObject::m_params.frameIdx <= 1 || IsDirty(kDirtySceneObjectChanged))
        {
            HighResolutionTimer frameTimer;
            
            dim3 blockSize, gridSize;
            KernelParamsFromImage(m_hostFrameBuffer, blockSize, gridSize);   
            
            const uint numSplatBlocks = (m_hostProjectedSplats->size() + 255) / 256;
            const uint numTileBlocks = (m_hostTileRanges->size() + 255) / 256;

            // Project the splats into camera space and assign keys
            Log::Warning("Projecting...");
            KernelProjectSplats<<< numSplatBlocks, 256>>>(cu_deviceInstance);

            Log::Warning("Computing overlapping tiles...");
            if (ComputeTileOverlaps())
            {
                // Sort the splats based on their keys
                Log::Warning("Sorting...");
                SortSplats();

                IsOk(cudaDeviceSynchronize());

                // Determine the ranges for the sorted values for each tile
                Log::Warning("Finding ranges...");
                KernelClearTileRanges << < numTileBlocks, 256 >> > (cu_deviceInstance);

                IsOk(cudaDeviceSynchronize());

                const uint numRefBlocks = (m_hostNumSplatRefs + 255) / 256;
                KernelDetermineTileRanges << < numRefBlocks, 256 >> > (cu_deviceInstance);

                //KernelVerify<< <1, 1 >> > (cu_deviceInstance);

                IsOk(cudaDeviceSynchronize());

                // Accumulate the frame
                Log::Warning("Rendering splats...");
                KernelRenderSortedSplats << < gridSize, blockSize >> > (cu_deviceInstance);
                //KernelRenderUnsortedSplats << < gridSize, blockSize >> > (cu_deviceInstance);

                //KernelDebug << <1, 1 >> > (cu_deviceInstance, 2u);

                IsOk(cudaDeviceSynchronize());

                Log::Debug("SplatRasteriser::Render(): took %.2fms", frameTimer.Get() * 1e3f);

                //Log::Warning("Rendering tiles...");
                //KernelRenderSplatTiles << < gridSize, blockSize >>> (cu_deviceInstance);

                //IsOk(cudaDeviceSynchronize());
            }         
        }

        // Denoise if necessary
        /*if (m_params.frameIdx % 500 == 0)
        {
            KernelDenoise << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);
        }*/

        IsOk(cudaDeviceSynchronize());

        // If there's no user interaction, signal the viewport to update intermittently to save compute
        constexpr float kViewportUpdateInterval = 1. / 2.f;
        if (m_redrawTimer.Get() > kViewportUpdateInterval)
        {
            SignalDirty(kDirtyViewportRedraw);
            m_redrawTimer.Reset();
        }

        /*if (m_renderTimer.Get() > 1.)
        {
            Log::Debug("Frame: %i", RenderableObject::m_params.frameIdx);
            m_renderTimer.Reset();
        }*/
    }

    __host__ void Host::SplatRasteriser::Clear()
    {
        m_hostFrameBuffer->Clear(vec4(0.f));

        RenderableObject::m_params.frameIdx = 0;
        Synchronise(kSyncParams);
    }

    __host__ bool Host::SplatRasteriser::OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject)
    {
        if (stateID == "kCreateDrawableObjectOpen" || stateID == "kCreateDrawableObjectHover")
        {
            m_isConstructed = true;
            m_isFinalised = true;
            if (stateID == "kCreateDrawableObjectOpen") { Log::Success("Opened path tracer %s", GetAssetID()); }

            return true;
        }
        else if (stateID == "kCreateDrawableObjectAppend")
        {
            m_isFinalised = true;
            return true;
        }

        return false;
    }

    __host__ bool Host::SplatRasteriser::OnRebuildDrawableObject()
    {
        /*m_scene = m_componentContainer->GenericObjects().FindFirstOfType<Host::SceneContainer>();
        if (!m_scene)
        {
            Log::Warning("Warning: path tracer '%s' expected an initialised scene container but none was found.");
        }*/
        
        return true;
    }

    __host__ bool Host::SplatRasteriser::IsClickablePoint(const UIViewCtx& viewCtx) const
    {
        return GetWorldSpaceBoundingBox().Contains(viewCtx.mousePos);
    }   

    __host__ BBox2f Host::SplatRasteriser::ComputeObjectSpaceBoundingBox()
    {
        return m_params.viewport.objectBounds;
    }

    __host__ bool Host::SplatRasteriser::Serialise(Json::Node& node, const int flags) const
    {
        DrawableObject::Serialise(node, flags);

        Json::Node lookNode = node.AddChildObject("viewport");
        lookNode.AddVector("dims", m_params.viewport.dims);

        return true;
    }

    __host__ bool Host::SplatRasteriser::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = DrawableObject::Deserialise(node, flags);
        
        Json::Node viewportNode = node.GetChildObject("viewport", flags);
        if (viewportNode)
        {
            isDirty |= viewportNode.GetVector("dims", m_params.viewport.dims, flags);
        }

        if (isDirty)
        {
            SignalDirty({ kDirtyParams });
        }

        return isDirty;
    }

}