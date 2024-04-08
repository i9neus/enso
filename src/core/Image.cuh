#pragma once

#include "math/Math.cuh"
#include "math/ColourUtils.cuh"
#include "AssetAllocator.cuh"

//#define CudaImageBoundCheck

namespace Enso
{		
	enum ImageAccessSignal : unsigned int { kImageUnlocked, kImageReadLocked, kImageWriteLocked };

	namespace Host { template<typename T> class Image; }

	namespace Device
	{				
		template<typename T>
		class Image : public Device::Asset, public AssetTags<Host::Image<T>, Device::Image<T>>
		{
			template<typename T> friend class Host::Image;
		public:
			__device__ Image(const uint width, const uint height, T* data) :
				m_width(width), m_height(height), cu_data(data), m_accessSignal(kImageUnlocked) {}
			//__device__ ~Image() = default; // NOTE: Commented out to suppress nvcc compiler warnings

			__host__ __device__ __forceinline__ unsigned int GetArea() const { return m_width * m_height; }
			__host__ __device__ __forceinline__ unsigned int GetMemorySize() const { return m_width * m_height * sizeof(T); }
			__host__ __device__ __forceinline__ unsigned int Width() const { return m_width; }
			__host__ __device__ __forceinline__ unsigned int Height() const { return m_height; }
			__host__ __device__ __forceinline__ ivec2 Dimensions() const { return ivec2(m_width, m_height); }
			__host__ __device__ __forceinline__ bool IsValid(const ivec2& xy) const { return xy.x >= 0 && xy.x < m_width&& xy.y >= 0 && xy.y < m_height; }

			__device__ __forceinline__ T* GetData() { return cu_data; }
			__device__ __forceinline__ unsigned int* AccessSignal() { return &m_accessSignal; }

			__device__ __forceinline__ T& At(int x, int y)
			{
#ifdef CudaImageBoundCheck
				if (x < 0 || x >= m_width || y < 0 || y >= m_height) { return nullptr; }
#endif
				return cu_data[y * m_width + x];
			}

			__device__ __forceinline__ T& At(const ivec2& xy)
			{
#ifdef CudaImageBoundCheck
				if (xy.x < 0 || xy.x >= m_width || xy.y < 0 || xy.y >= m_height) { return nullptr; }
#endif
				return cu_data[xy.y * m_width + xy.x];
			}

			__device__ __forceinline__ typename std::enable_if<std::is_same<T, vec4>::value>::type BlendPixel(const ivec2& xy, const vec4& rgba)
			{
#ifdef CudaImageBoundCheck
				if (xy.x < 0 || xy.x >= m_width || xy.y < 0 || xy.y >= m_height) { return nullptr; }
#endif
				auto& pixel = cu_data[xy.y * m_width + xy.x];
				pixel = Blend(pixel, rgba);
			}

			__device__ __inline__ T Lerp(const vec2& xy)
			{
				if (xy.x < 0. || xy.x >= m_width - 1 || xy.y < 0. || xy.y >= m_height - 1) { return T(0); }

				float dx = fract(xy.x), dy = fract(xy.y);
				int ix = int(xy.x), iy = int(xy.y);
				
				return (cu_data[iy  * m_width + ix] * (1.0 - dx) + cu_data[iy * m_width + (1 + ix)] * dx) * (1 - dy) +
					   (cu_data[(iy + 1) * m_width + ix] * (1.0 - dx) + cu_data[(iy + 1) * m_width + (1 + ix)] * dx) * dy;
			}

			__device__ T Texture(vec2 uv)
			{
				uv *= vec2(m_width, m_height);
				int iu = int(uv.x), iv = int(uv.y);
				float du, dv;

				// Clip to the bounds of the texture
				if (iu < 0) { iu = 0; du = 0; }
				else if (iu >= m_width - 1) { iu = m_width - 2; du = 1.0f; }
				else { du = fract(uv.x); }
				if (iv < 0) { iv = 0; dv = 0; }
				else if (iv >= m_height - 1) { iv = m_height - 2; dv = 1.0f; }
				else { dv = fract(uv.y); }

				return (cu_data[iv * m_width + iu] * (1.0 - du) + cu_data[iv * m_width + (1 + iu)] * du) * (1 - dv) +
					   (cu_data[(iv + 1) * m_width + iu] * (1.0 - du) + cu_data[(iv + 1) * m_width + (1 + iu)] * du) * dv;
			}

			template<typename = typename std::enable_if<std::is_same<T, vec4>::value>>
			__device__ __forceinline__ void Accumulate(const ivec2& xy, const vec3& value, const bool isAlive)
			{
				auto& texel = cu_data[xy.y * m_width + xy.x];
				texel.xyz += value;
				if (!isAlive) 
				{
					texel.w = -(texel.w + 1.0f); 
				}
			}

			__device__ void Clear(const ivec2& xy, const T& value);

		protected:
			Image() : m_width(0), m_height(0), cu_data(nullptr), m_accessSignal(kImageUnlocked) {}

			unsigned int	m_width;
			unsigned int	m_height;
			T*				cu_data;
			unsigned int    m_accessSignal;
		};

		template class Image<vec4>;
		using ImageRGBW = Image<vec4>;
		using ImageRGBA = Image<vec4>;
	}

	namespace Host
	{
		template<typename T>
		class Image : public Host::Asset, public AssetTags<Host::Image<T>, Device::Image<T>>
		{
		protected:
			Device::Image<T>* cu_deviceData;
			Device::Image<T>  m_hostData;

			cudaStream_t	  m_hostStream;
			dim3			  m_block;
			dim3		      m_grid;

			AssetAllocator    m_allocator;

		public:
			using Pixel = T;

			__host__ Image(const std::string& id, unsigned int width, unsigned int height, cudaStream_t hostStream);
			__host__ virtual ~Image() { OnDestroyAsset(); }

			__host__ virtual void OnDestroyAsset() override final;

			__host__ cudaStream_t GetHostStream() const { return m_hostStream; }
			__host__ void SetHostStream(cudaStream_t hostStream) { m_hostStream = hostStream; }

			__host__ Device::Image<T>* GetDeviceInstance() const
			{
				AssertMsg(cu_deviceData, "Image has not been initialised!");
				return cu_deviceData;
			}
			__host__ inline const Device::Image<T>& GetMetadata() const { return m_hostData; }
			__host__ inline bool IsCreated() const { return cu_deviceData != nullptr; }
			__host__ inline dim3 GetBlockSize() const { return m_block; }
			__host__ inline dim3 GetGridSize() const { return m_grid; }
			__host__ void Download(std::vector<T>& rawData) const;

			__host__ void SignalChange(cudaStream_t hostStream, const unsigned int currentState, const unsigned int newState);
			__host__ inline void SignalSetRead(cudaStream_t hostStream = nullptr) { SignalChange(hostStream, kImageUnlocked, kImageReadLocked); }
			__host__ inline void SignalUnsetRead(cudaStream_t hostStream = nullptr) { SignalChange(hostStream, kImageReadLocked, kImageUnlocked); }
			__host__ inline void SignalSetWrite(cudaStream_t hostStream = nullptr) { SignalChange(hostStream, kImageUnlocked, kImageWriteLocked); }
			__host__ inline void SignalUnsetWrite(cudaStream_t hostStream = nullptr) { SignalChange(hostStream, kImageWriteLocked, kImageUnlocked); }
			__host__ void Clear(const T& value);
			__host__ void CopyImageToD3DTexture(unsigned int clientWidth, unsigned int clientHeight, cudaSurfaceObject_t cuSurface, cudaStream_t hostStream);
		};

		template class Image<vec4>;
		using ImageRGBW = Image<vec4>;
		using ImageRGBA = Image<vec4>;
	}
}