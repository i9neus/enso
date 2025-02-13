#pragma onceh

#include "../../CudaHeaders.cuh"
#include "../../math/Math.cuh"
#include "../../assets/Asset.cuh"
#include "../../containers/DualImage.cuh"
#include "../../containers/Object.cuh"

namespace Enso
{
    namespace Device
    {
        class PDF2 : public Device::Asset
        {
        private:
            Device::DualImage3f*         m_cdf = nullptr;
            int                          m_width = 0;  // NOTE: Width does not include the padding column. Should always be power of two.
            int                          m_height = 0;
            int                          m_area = 0;
            ivec2                        m_dims;

        public:
            __host__ __device__ PDF2() {}

            __device__ __forceinline__ int Width() const { return m_width; }
            __device__ __forceinline__ int Height() const { return m_height; }
            __device__ __forceinline__ int Area() const { return m_area; }
            __device__ __forceinline__ ivec2 Dims() const { return m_dims; }
            __device__ vec3 Sample(const vec2& xi, vec2& p) const;
            __device__ vec3 Evaluate(vec2 p) const;

            __device__ void Set(const int x, const int y, const float pdf, const float value);

            __device__ void Synchronise(Device::DualImage3f* cdf)
            {
                CudaAssert(cdf);
                m_cdf = cdf;
                m_width = cdf->Width() - 1;
                m_height = cdf->Height();
                m_area = m_width * m_height;
                m_dims = ivec2(m_width, m_height);
            }
        };
    }

    namespace Host
    {
        class PDF2 : public Host::Asset
        {
        private:
            Device::PDF2*                       cu_deviceInstance;
            Device::PDF2                        m_objects;
            AssetHandle<Host::DualImage3f>      m_cdf;
            Cuda::Object<vec2>                  m_colNorm;
            cudaStream_t                        m_cudaStream;

        public:
            __host__ PDF2(const Asset::InitCtx&, const int width, const int height, cudaStream_t stream = nullptr);
            __host__ virtual ~PDF2();

            __host__ PDF2(const PDF2&) = delete;
            __host__ PDF2(PDF2&&) = delete;
            __host__ PDF2& operator=(const PDF2&) = delete;
            __host__ PDF2& operator=(PDF2&&) = delete;

            __host__ void Rebuild();
            __host__ Device::PDF2* GetDeviceInstance() { return cu_deviceInstance; }
            __host__ Host::DualImage3f& GetCDF() { return *m_cdf; }

            __host__ int Width() const { return m_cdf->Width(); }
            __host__ int Height() const { return m_cdf->Height(); }
            __host__ int Area() const { return m_cdf->Area(); }

            __host__ float& At(const int x, const int y) 
            { 
                AssertMsgFmt(x >= 0 && x < m_cdf->Width() && y >= 0 && y <= m_cdf->Height(), "PDF coordinate [%i, %i) out of bounds.", x, y);
                return *m_cdf->At(x, y); 
            }
        };
    }
}