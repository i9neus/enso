#pragma once

#include "math/CudaMath.cuh"
#include "CudaRay.cuh"

//#define CudaImageBoundCheck

namespace Cuda
{		
	enum AccessSignal : unsigned int { kImageUnlocked, kImageReadLocked, kImageWriteLocked };

	namespace Device
	{		
		template<typename T>
		class Image
		{
		public:
			__host__ __device__ unsigned int GetArea() const { return m_width * m_height; }
			__host__ __device__ unsigned int GetMemorySize() const { return m_width * m_height * sizeof(float4); }
			__host__ __device__ unsigned int Width() const { return m_width; }
			__host__ __device__ unsigned int Height() const { return m_height; }

			__device__ T* GetData() { return cu_data; }
			__device__ unsigned int* AccessSignal() { return &m_accessSignal; }

			__device__ T& At(int x, int y)
			{
#ifdef CudaImageBoundCheck
				if (x < 0 || x >= m_width || y < 0 || y >= m_height) { return nullptr; }
#endif
				return cu_data[y * m_height + x];
			}

			__device__ T& At(const ivec2& xy)
			{
#ifdef CudaImageBoundCheck
				if (xy.x < 0 || xy.x >= m_width || xy.y < 0 || xy.y >= m_height) { return nullptr; }
#endif
				return cu_data[xy.y * m_height + xy.x];
			}

			__device__ void Clear(const ivec2& xy, const T& value);

		protected:
			Image() : m_width(0), m_height(0), cu_data(nullptr), m_accessSignal(kImageUnlocked) {}
			~Image() = default;

			unsigned int	m_width;
			unsigned int	m_height;
			T* cu_data;
			unsigned int    m_accessSignal;
		};

		template class Image<vec4>;
		using ImageRGBW = Image<vec4>;
		using ImageRGBA = Image<vec4>;

		template class Image<PackedRay>;
		using PackedRayBuffer = Image<PackedRay>;
	}

	namespace Host
	{
		template<typename T>
		class Image : public Device::Image<T>, public AssetBase
		{
		protected:
			Device::Image<T>* cu_deviceImage;
			cudaStream_t	  m_hostStream;
			dim3			  m_block;
			dim3		      m_grid;

		public:
			using Pixel = T;

			Image(unsigned int width, unsigned int height, cudaStream_t hostStream);
			~Image() { OnDestroyAsset(); }

			__host__ virtual void OnDestroyAsset() override final;

			__host__ cudaStream_t GetHostStream() const { return m_hostStream; }
			__host__ Device::Image<T>* GetDeviceImage() const
			{
				AssertMsg(cu_deviceImage, "Image has not been initialised!");
				return cu_deviceImage;
			}
			__host__ bool IsCreated() const { return cu_deviceImage != nullptr; }

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

		template class Image<PackedRay>;
		using PackedRayBuffer = Image<PackedRay>;
	}
}