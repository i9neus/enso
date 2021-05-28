#include "CudaImage.cuh"

namespace Cuda
{
	template<typename T>
	__global__ void KernelSignalChange(Device::Image<T>* image, const unsigned int currentState, const unsigned int newState) { atomicCAS(image->AccessSignal(), currentState, newState); }

	template<typename T>
	__global__ void KernelClear(Device::Image<T>* image, const T& value) 
	{ 
		//if (*(image->AccessSignal()) != kImageWriteLocked) { return; }

		const uint kx = blockIdx.x * blockDim.x + threadIdx.x;
		const uint ky = blockIdx.y * blockDim.y + threadIdx.y;
		if (kx < image->Width() && ky < image->Height())
		{
			image->At(ky, ky) = value;
		}
		
		//image->Clear(KERNEL_COORDS_IVEC2, value);
	}
	
	template<typename T>
	__device__ void Device::Image<T>::Clear(const ivec2& xy, const T& value)
	{
		if (xy.x < m_width && xy.y < m_height)
		{
			At(xy.x, xy.y) = value;
		}
	}

	template<typename T>
	__global__ void KernelCopyImageToD3DTexture(unsigned int clientWidth, unsigned int clientHeight, Device::Image<T>* image, cudaSurfaceObject_t cuSurface)
	{
		if (*(image->AccessSignal()) != kImageReadLocked) { return; }

		unsigned int kx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int ky = blockIdx.y * blockDim.y + threadIdx.y;

		if (kx >= clientWidth || ky >= clientHeight) { return; }

		int px = kx - clientWidth / 2 + image->Width() / 2;
		int py = ky - clientHeight / 2 + image->Height() / 2;

		if (px < 0 || px >= image->Width() || py < 0 || py >= image->Height()) { return; }

		surf2Dwrite(*reinterpret_cast<float4*>(&(image->At(px, py))), cuSurface, kx * 16, ky);
	}

	template<typename T>
	__host__ Host::Image<T>::Image(unsigned int width, unsigned int height, cudaStream_t hostStream)
	{
		m_width = width;
		m_height = height;
		m_hostStream = hostStream;
		m_accessSignal = kImageUnlocked;

		m_block = dim3(16, 16, 1);
		m_grid = dim3((m_width + 15) / 16, (m_height + 15) / 16, 1);

		checkCudaErrors(cudaMalloc((void**)&cu_data, sizeof(T) * width * height));
		checkCudaErrors(cudaMalloc((void**)&cu_deviceImage, sizeof(Device::Image<T>)));
		checkCudaErrors(cudaMemcpy(cu_deviceImage, static_cast<Device::Image<T>*>(this), sizeof(Device::Image<T>), cudaMemcpyHostToDevice));
	}

	template<typename T>
	__host__ void Host::Image<T>::SignalChange(cudaStream_t otherStream, const unsigned int currentState, const unsigned int newState)
	{ 
		cudaStream_t hostStream = otherStream ? otherStream : m_hostStream;
		KernelSignalChange << < 1, 1, 0, hostStream >> > (cu_deviceImage, currentState, newState);
	}	

	template<typename T>
	__host__ void Host::Image<T>::Clear(const T& value) 
	{ 
		KernelClear << < m_grid, m_block, 0, m_hostStream >> > (cu_deviceImage, value);
	}

	template<typename T>
	__host__ void Host::Image<T>::OnDestroyAsset()
	{
		SafeFreeDeviceMemory(&cu_data);
		SafeFreeDeviceMemory(&cu_deviceImage);
	}

	// The host CPU Sinewave thread spawner
	template<typename T>
	__host__ void Host::Image<T>::CopyImageToD3DTexture(unsigned int clientWidth, unsigned int clientHeight, cudaSurfaceObject_t cuSurface, cudaStream_t hostStream)
	{		
		dim3 block(16, 16, 1);
		dim3 grid((clientWidth + 15) / 16, (clientHeight + 15) / 16, 1);

		SignalSetRead(hostStream);
		KernelCopyImageToD3DTexture << < grid, block, 0, hostStream >> > (clientWidth, clientHeight, cu_deviceImage, cuSurface);
		SignalUnsetRead(hostStream);

		getLastCudaError("CopyImageToD3DTexture execution failed.\n");
	}
}