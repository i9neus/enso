#pragma once

#include "core/math/Math.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/assets/AssetAllocator.cuh"

#include <functional>
#include <thread>
#include <vector>

namespace Enso
{
    struct DualImageRect
    {
        int x0, y0, x1, y1;

        DualImageRect() : x0(std::numeric_limits<int>::max()), y0(std::numeric_limits<int>::max()), x1(std::numeric_limits<int>::min()), y1(std::numeric_limits<int>::min()) {}
        DualImageRect(int _x0, int _y0, int _x1, int _y1) : x0(_x0), y0(_y0), x1(_x1), y1(_y1) {}

        inline int Area() const { return (x1 - x0) * (y1 - y0); }
        inline int Width() const { return x1 - x0; }
        inline int Height() const { return y1 - y0; }
        inline operator bool() const { return x1 > x0 && y1 > y0; }
        inline bool Contains(const int x, const int y) const { return x >= x0 && x < x1&& y >= y0 && y < y1; }
    };

    enum DualImageFlags : int { kDualImageNearest, kDualImageBilinear };

    inline DualImageRect Intersection(const DualImageRect& a, const DualImageRect& b)
    {
        return DualImageRect(std::max(a.x0, b.x0), std::max(a.y0, b.y0), std::min(a.x1, b.x1), std::min(a.y1, b.y1));
    }

    namespace Host { template<typename, int> class DualImage; }

    namespace Device
    {
        template<typename Type, int Channels>
        class DualImage
        {
            template<typename T, int C> friend class Host::DualImage;

        public:
            __host__ __device__ DualImage() {}

            __host__ __device__ void Synchronise(const DualImage& data) { *this = data; }
            __host__ __device__ void Validate() const
            {
                CudaAssert((m_data && m_width > 0 && m_height > 0 && m_area > 0) || (!m_data && m_width == 0 && m_height == 0 && m_area == 0));
            }

            __host__ __device__ bool Contains(const int x, const int y) const { return x >= 0 && x < m_width && y >= 0 && y < m_height; }

            __host__ __device__ __forceinline__ int Width() const { return m_width; }
            __host__ __device__ __forceinline__ int Height() const { return m_height; }
            __host__ __device__ __forceinline__ int Area() const { return m_area; }
            __host__ __device__ __forceinline__ ivec2 Dimensions() const { return ivec2(m_width, m_height); }
            __host__ __device__ __forceinline__ int Size() const { return m_area * Channels; }
            __host__ __device__ __forceinline__ DualImageRect Rect() const { return DualImageRect(0, 0, m_width, m_height); }

            __host__ __device__ __forceinline__ void Set(const int x, const int y, const float* data) { memcpy(&m_data[(y * m_width + x) * Channels], data, sizeof(Type) * Channels); }

            __host__ __device__ __forceinline__ Type* operator()(const int x, const int y) { return &m_data[(y * m_width + x) * Channels]; }
            __host__ __device__ __forceinline__ const Type* operator()(const int x, const int y) const { return &m_data[(y * m_width + x) * Channels]; }

            __host__ __device__ __forceinline__ Type* At(const int x, const int y) { return &m_data[(y * m_width + x) * Channels]; }
            __host__ __device__ __forceinline__ const Type* At(const int x, const int y) const { return &m_data[(y * m_width + x) * Channels]; }
            __host__ __device__ __forceinline__ Type* At(const ivec2& p) { return &m_data[(p.y * m_width + p.x) * Channels]; }
            __host__ __device__ __forceinline__ const Type* At(const ivec2& p) const { return &m_data[(p.y * m_width + p.x) * Channels]; }   
            
            template<typename CastT> __host__ __device__ __forceinline__ CastT& As(const int x, const int y) { return *reinterpret_cast<CastT*>(&m_data[(y * m_width + x) * Channels]); }
            template<typename CastT> __host__ __device__ __forceinline__ const CastT& As(const int x, const int y) const { return *reinterpret_cast<const CastT*>(&m_data[(y * m_width + x) * Channels]); }
            template<typename CastT> __host__ __device__ __forceinline__ CastT& As(const ivec2& p) { return *reinterpret_cast<CastT*>(&m_data[(p.y * m_width + p.x) * Channels]); }
            template<typename CastT> __host__ __device__ __forceinline__ const CastT& As(const ivec2& p) const { return *reinterpret_cast<const CastT*>(&m_data[(p.y * m_width + p.x) * Channels]); }

            __host__ __device__ __forceinline__ Type& operator[](const int i) { return m_data[i]; }
            __host__ __device__ __forceinline__ Type operator[](const int i) const { return m_data[i]; }

            template<int InterpolationType>
            __host__ __device__ void Sample(float u, float v, Type* pixel) const
            {
                if (InterpolationType == kDualImageBilinear)
                {
                    // For interpolation, we assume that the pixel values are defined at the mid-point of each logcal pixel
                    // and that the values are clamped at the boundaries.
                    //  
                    // Example: for a 2 pixel image in 1 dimensions, values p[0] and p[1] correspond to coordinates 0.25 and 0.75 respectively
                    // 
                    // 0.0      0.5       1.0
                    //  |   *    |    *    |
                    //     p[0]     p[1]

                    int iu, iv;
                    float du, dv;
                    u = std::max(0.f, u * m_width - 0.5f);
                    v = std::max(0.f, v * m_height - 0.5f);
                    if (u >= m_width - 1) { iu = m_width - 2; du = 1; }
                    else { iu = int(u); du = fract(u); }
                    if (v >= m_height - 1) { iv = m_height - 2; dv = 1; }
                    else { iv = int(v); dv = fract(v); }

                    int idx = (iv * m_width + iu) * Channels;
                    for (int c = 0; c < Channels; ++c)
                    {
                        const Type t00 = m_data[idx + c];
                        const Type t10 = m_data[idx + Channels + c];
                        const Type t01 = m_data[idx + m_width * Channels + c];
                        const Type t11 = m_data[idx + (m_width + 1) * Channels + c];
                        pixel[c] = mix(mix(t00, t10, du), mix(t01, t11, du), dv);
                    }
                }
                if (InterpolationType == kDualImageNearest)
                {
                    int idx = Channels * (clamp(int(v * m_height - 0.5), 0, m_height - 1) * m_width +
                        clamp(int(u * m_width - 0.5), 0, m_width - 1));

                    for (int c = 0; c < Channels; ++c, ++idx)
                    {
                        pixel[c] = m_data[idx];
                    }
                }
            }

            template<int InterpolationType>
            __host__ __device__ inline Type Sample(float u, float v) const
            {
                Type pixel[Channels];
                Sample<InterpolationType>(u, v, pixel);
                return pixel[0];
            }

            __host__ __device__ inline void Sample(int x, int y, Type* pixel) const
            {
                const int idx = Channels * (clamp(y, 0, m_height - 1) * m_width + clamp(x, 0, m_width - 1));
                for (int c = 0; c < Channels; ++c, ++idx)
                {
                    pixel[c] = m_data[idx];
                }
            }

            __host__ __device__ inline Type Sample(int x, int y) const
            {
                return m_data[Channels * (clamp(y, 0, m_height - 1) * m_width + clamp(x, 0, m_width - 1))];
            }

        protected:            
            Type*               m_data = nullptr;
            int                 m_width = 0;
            int                 m_height = 0;
            int                 m_area = 0;
        };

        using DualImage4f = DualImage<float, 4>;
        using DualImage3f = DualImage<float, 3>;
        using DualImage2f = DualImage<float, 2>;
        using DualImage1f = DualImage<float, 1>;
    }

    namespace Host
    {
        template<typename Type, int Channels>
        class DualImage : public Device::DualImage<Type, Channels>, public Host::Asset
        {
        private:
            using DeviceType = Device::DualImage<Type, Channels>;

            DeviceType               m_deviceData;
            DeviceType*              cu_deviceInstance = nullptr;

        public:
            using MapFunctor = std::function<void(int, int, Type*)>;
            using ParallelMapFunctor = std::function<void(int, int, int, Type*)>;

        public:
            __host__ DualImage(const Asset::InitCtx& initCtx) :
                Host::Asset(initCtx)
            {
                cu_deviceInstance = AssetAllocator::InstantiateOnDevice<DeviceType>(*this);
            }

            __host__ DualImage(const Asset::InitCtx& initCtx, const int width, const int height, const Type* data = nullptr) : DualImage(initCtx)
            {
                Resize(width, height);
                if (data)
                {
                    memcpy(m_vector.data(), data, sizeof(Type) * Channels * m_area);
                }
                Upload();
            }

            __host__ ~DualImage()
            {
                AssetAllocator::GuardedFreeDevice1DArray(*this, m_area * Channels, &m_deviceData.m_data);
                AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
            }
            __host__ DualImage(const DualImage& other) = delete;
            __host__ DualImage(DualImage&& other) = delete;
            __host__ DualImage& operator=(const DualImage& other) = delete;
            __host__ DualImage& operator=(DualImage&& other) = delete;

            __host__ inline DeviceType* GetDeviceInstance() { return cu_deviceInstance; }
            __host__ inline const DeviceType* GetDeviceInstance() const { return cu_deviceInstance; }

            __host__ void Resize(const int width, const int height)
            {
                if (m_width == width && m_height == height) { return; }

                AssertMsgFmt(width >= 0 && height >= 0, "Invalid image dimensions: %i x %i", width, height);

                // Reallocate the device memory
                AssetAllocator::GuardedFreeDevice1DArray(*this, m_area * Channels, &m_deviceData.m_data);
                if (width * height != 0)
                {
                    AssetAllocator::GuardedAllocDevice1DArray(*this, width * height * Channels, &m_deviceData.m_data, 0u);
                }      
                m_deviceData.m_width = width;
                m_deviceData.m_height = height;
                m_deviceData.m_area = width * height;

                // Reallocate the host memory
                m_width = width;
                m_height = height;
                m_area = width * height;
                m_vector.resize(m_area * Channels, Type(0));
                m_data = m_vector.data();
            }

            template<int OtherChannels>
            __host__ inline void ResizeFrom(const DualImage<Type, OtherChannels>& other) { Resize(other.Width(), other.Height()); }

            __host__ DualImage<Type, 1> ExtractChannel(const int chnlIdx) const
            {
                DualImage<Type, 1> chnlData(m_width, m_height);
                for (int i = 0; i < m_area; ++i)
                {
                    chnlData[i] = m_vector[i * Channels + chnlIdx];
                }
                return chnlData;
            }

            __host__ DualImage<Type, 1> ExtractLuminance() const
            {
                static_assert(Channels == 3, "Extract luminance requires a 3-channel RGB image.");
                DualImage<Type, 1> lum(m_width, m_height);
                for (int i = 0, j = 0; i < m_area; ++i, j += 3)
                {
                    Assert(i < lum.Vector().size());
                    lum[i] = m_vector[j] * 0.17691 + m_vector[j + 1] * 0.8124 + m_vector[j + 2] * 0.01063;
                }
                return lum;
            }

            __host__ void EmplaceChannel(const DualImage<Type, 1>& chnlData, const int chnlIdx)
            {
                AssertFmt(chnlData.Width() == m_width && chnlData.Height() == m_height, "Size mismatch!");
                for (int i = 0; i < m_area; ++i)
                {
                    m_vector[i * Channels + chnlIdx] = chnlData[i];
                }
            }

            __host__ void ApplyGamma(const float gamma)
            {
                if (gamma != 1)
                {
                    for (auto& p : m_vector)
                    {
                        p = std::pow(p, gamma);
                    }
                }
            }

            // Sets all pixels in the image to zero
            __host__ void Erase() { std::memset(m_vector.data(), 0, sizeof(Type) * m_vector.size()); }

            __host__ void Saturate() { for (auto& p : m_vector) { p = saturate(p); } }

            __host__ void Clamp(const Type lower, const Type upper) { for (auto& p : m_vector) { p = clamp(p, lower, upper); } }

            __host__ Type* Data() { return m_vector.data(); }
            __host__ const Type* Data() const { return m_vector.data(); }
            __host__ std::vector<Type>& Vector() { return m_vector; }
            __host__ const std::vector<Type> Vector() const { return m_vector; }

            __host__ int GetThreadCount(const int maxThreads = 16) const
            {
                int numThreads = std::max(1, int(std::thread::hardware_concurrency()));
                if (maxThreads > 0)
                {
                    numThreads = std::min(maxThreads, numThreads);
                }
                return numThreads;
            }

            __host__ void ParallelMap(ParallelMapFunctor setPixel, DualImageRect region = DualImageRect(), const int maxThreads = 16)
            {
                // If no region was specified, reinitialise it to the entire image
                if (!region) { region = DualImageRect(0, 0, m_width, m_height); }

                const int numThreads = GetThreadCount(maxThreads);
                const int regionArea = region.Area();

                // Launch the worker threads
                std::vector<std::thread> workers;
                for (int i = 0; i < numThreads; ++i)
                {
                    const int startPixel = i * regionArea / numThreads;
                    const int endPixel = (i + 1) * regionArea / numThreads;
                    workers.emplace_back(&DualImage<Type, Channels>::MapThread, this, region, startPixel, endPixel, i, setPixel);
                }

                // Wait for all the workers to finish
                for (int i = 0; i < numThreads; ++i) { workers[i].join(); }
            }

            __host__ void Map(MapFunctor setPixel, DualImageRect region = DualImageRect(), const int maxThreads = 16)
            {
                // If no region was specified, reinitialise it to the entire image
                if (!region) { region = DualImageRect(0, 0, m_width, m_height); }

                for (int y = region.y0; y < region.y1; ++y)
                {
                    for (int x = region.x0; x < region.x1; ++x)
                    {
                        setPixel(x, y, &m_vector[Channels * (y * m_width + x)]);
                    }
                }
            }

            __host__ void Download()
            {
                IsOk(cudaMemcpy(m_data, m_deviceData.m_data, sizeof(Type) * m_area, cudaMemcpyDeviceToHost));
            }

            __host__ void Upload()
            {
                IsOk(cudaMemcpy(m_deviceData.m_data, m_data, sizeof(Type) * m_area, cudaMemcpyHostToDevice));
                
                SynchroniseObjects<DeviceType>(cu_deviceInstance, m_deviceData);
            }

            __host__ std::pair<dim3, dim3> GetKernelParams(const size_t blockSize = 16) const
            {
                return { dim3(blockSize, blockSize, 1),
                         dim3((m_width + (blockSize - 1)) / blockSize, (m_height + (blockSize - 1)) / blockSize, 1) };
            }

        private:
            __host__ void MapThread(const DualImageRect& region, const int startPixel, const int endPixel, const int threadIdx, ParallelMapFunctor setPixel)
            {
                for (int i = startPixel; i < endPixel; ++i)
                {
                    const int x = region.x0 + i % region.Width();
                    const int y = region.y0 + i / region.Width();
                    setPixel(x, y, threadIdx, &m_vector[Channels * (y * m_width + x)]);
                }
            }

            __host__ void CopyAttribs(const DualImage& other)
            {
                m_width = other.m_width;
                m_height = other.m_height;
                m_area = other.m_area;
            }

        private:
            std::vector<Type>   m_vector;
            /*int                 m_width;
            int                 m_height;
            int                 m_area;*/
        };

        using DualImage4f = DualImage<float, 4>;
        using DualImage3f = DualImage<float, 3>;
        using DualImage2f = DualImage<float, 2>;
        using DualImage1f = DualImage<float, 1>;
    }
}