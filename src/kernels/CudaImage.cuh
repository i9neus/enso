#pragma once

#include "math/CudaMath.cuh"

//#define CudaImageBoundCheck

namespace Cuda
{		
	__host__ __device__ inline float4 operator+(const float4& lhs, const float4& rhs) {
		return lhs;
	}
	
	class DeviceImage
	{
	public: 
		enum AccessSignal : unsigned int { kUnlocked, kReadLocked, kWriteLocked };

		__host__ __device__ unsigned int GetArea() const { return m_width * m_height; }
		__host__ __device__ unsigned int GetMemorySize() const { return m_width * m_height * sizeof(float4); }
		__host__ __device__ unsigned int Width() const { return m_width; }
		__host__ __device__ unsigned int Height() const { return m_height; }

		__device__ float4* GetData() { return cu_data; }
		__device__ unsigned int* AccessSignal() { return &m_accessSignal; }

		__device__ float4* At(int x, int y)
		{
#ifdef CudaImageBoundCheck
			if (x < 0 || x >= m_width || y < 0 || y >= m_height) { return nullptr; }
#endif
			return &cu_data[y * m_height + x];
		}

	protected:
		friend class	HostImage;

		DeviceImage() : m_width(0), m_height(0), cu_data(nullptr), m_accessSignal(AccessSignal::kUnlocked) {}
		DeviceImage(const unsigned int width, const unsigned int height, float4* data) : m_width(width), m_height(height), cu_data(data) {}
		~DeviceImage() = default;

		unsigned int	m_width;
		unsigned int	m_height;
		float4*			cu_data;
		unsigned int    m_accessSignal;
	};

	class HostImage : public DeviceImage
	{
	protected:
		DeviceImage*	cu_deviceImage;
		cudaStream_t    m_hostStream;

	public: 
		HostImage() : cu_deviceImage(nullptr) {}
		~HostImage() { Destroy(); }

		__host__ void Create(unsigned int width, unsigned int height, cudaStream_t hostStream);
		__host__ void Destroy();
		__host__ cudaStream_t GetHostStream() const { return m_hostStream; }
		__host__ DeviceImage* GetDeviceImage() const 
		{ 
			AssertMsg(cu_deviceImage, "Image has not been initialised!");
			return cu_deviceImage; 
		}

		__host__ void SignalSetRead(cudaStream_t hostStream);
		__host__ void SignalUnsetRead(cudaStream_t hostStream);
		__host__ void SignalSetWrite(cudaStream_t hostStream);
		__host__ void SignalUnsetWrite(cudaStream_t hostStream);
		__host__ void Clear();
		__host__ void CopyImageToD3DTexture(unsigned int clientWidth, unsigned int clientHeight, cudaSurfaceObject_t cuSurface, cudaStream_t hostStream);

	};
}