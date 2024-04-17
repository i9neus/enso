#include "Image.cuh"

namespace Enso
{
	template<typename T>
	__global__ void KernelSignalChange(Device::Image<T>* image, const unsigned int currentState, const unsigned int newState) { atomicCAS(image->AccessSignal(), currentState, newState); }

	template<typename T>
	__global__ void KernelClear(Device::Image<T>* image, const T value) 
	{ 
		//if (*(image->AccessSignal()) != kImageWriteLocked) { return; }
		
		image->Clear(kKernelPos<ivec2>(), value);
	}

	template<typename T>
	__host__ Host::Image<T>::~Image()
	{
		m_allocator.DestroyOnDevice(cu_deviceData);
		m_allocator.GuardedFreeDeviceArray(m_hostData.m_width * m_hostData.m_height, &m_hostData.cu_data);
	}
	
	template<typename T>
	__device__ void Device::Image<T>::Clear(const ivec2& xy, const T& value)
	{
		if(IsValid(xy))
		{
			At(xy) = value;
		}
	}

	template<typename T>
	__global__ void KernelCopyImageToD3DTexture(unsigned int clientWidth, unsigned int clientHeight, Device::Image<T>* image, cudaSurfaceObject_t cuSurface)
	{
		//if (*(image->AccessSignal()) != kImageReadLocked) { return; }

		unsigned int kx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int ky = blockIdx.y * blockDim.y + threadIdx.y;

		if (kx >= clientWidth || ky >= clientHeight) { return; }

		int px = kx - clientWidth / 2 + image->Width() / 2;
		int py = ky - clientHeight / 2 + image->Height() / 2;

		if (px < 0 || px >= image->Width() || py < 0 || py >= image->Height()) { return; }

		// Alpha blend rgba values onto a black background and set w to full opacity
		vec4 rgba = image->At(px, py);
		rgba.xyz *= rgba.w;
		rgba.w = 1.0f;

		surf2Dwrite(*reinterpret_cast<float4*>(&rgba), cuSurface, kx * 16, ky);
	}

	template<typename T>
	__host__ Host::Image<T>::Image(const std::string& id, unsigned int width, unsigned int height, cudaStream_t hostStream) :
		Asset(id),
		m_allocator(*this),
		cu_deviceData(nullptr)
	{		
		// Prepare the host data
		m_hostData.m_width = width;
		m_hostData.m_height = height;
		m_hostData.m_accessSignal = kImageUnlocked;

		m_allocator.GuardedAllocDeviceArray(width * height, &m_hostData.cu_data);

		cu_deviceData = m_allocator.InstantiateOnDevice<Device::Image<T>>(width, height, m_hostData.cu_data);

		m_hostStream = hostStream;
		m_block = dim3(16, 16, 1);
		m_grid = dim3((width + 15) / 16, (height + 15) / 16, 1);
	}

	template<typename T>
	__host__ void Host::Image<T>::SignalChange(cudaStream_t otherStream, const unsigned int currentState, const unsigned int newState)
	{ 
		cudaStream_t hostStream = otherStream ? otherStream : m_hostStream;
		KernelSignalChange << < 1, 1, 0, hostStream >> > (cu_deviceData, currentState, newState);
	}	

	template<typename T>
	__host__ void Host::Image<T>::Clear(const T& value)
	{ 
		KernelClear << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceData, value);
	}

	template<typename T>
	__host__ void Host::Image<T>::CopyImageToD3DTexture(unsigned int clientWidth, unsigned int clientHeight, cudaSurfaceObject_t cuSurface, cudaStream_t hostStream)
	{		
		dim3 block(16, 16, 1);
		dim3 grid((clientWidth + 15) / 16, (clientHeight + 15) / 16, 1);
		
		SignalSetRead(hostStream);
		KernelCopyImageToD3DTexture << < grid, block, 0, hostStream >> > (clientWidth, clientHeight, cu_deviceData, cuSurface);
		SignalUnsetRead(hostStream);

		getLastCudaError("CopyImageToD3DTexture execution failed.\n");
	}

	template<typename T>
	__host__ void Host::Image<T>::Download(std::vector<T>& rawData) const
	{
		Assert(m_hostData.cu_data);
		if (m_hostData.GetArea() == 0) { return; }

		rawData.resize(m_hostData.GetArea());

		IsOk(cudaMemcpy(rawData.data(), m_hostData.cu_data, sizeof(T) * m_hostData.GetArea(), cudaMemcpyDeviceToHost));
	}
}