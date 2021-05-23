#include "CudaImage.cuh"

namespace Cuda
{
	__global__ void SignalChange(DeviceImage* image, const unsigned int currentState, const unsigned int newState) { atomicCAS(image->AccessSignal(), currentState, newState); }
	__global__ void Clear(DeviceImage* image) { memset(image->GetData(), 0, image->GetMemorySize()); }
	
	__host__ void HostImage::SignalSetRead(cudaStream_t hostStream) { SignalChange << < 1, 1, 0, hostStream >> > (cu_deviceImage, DeviceImage::kUnlocked, DeviceImage::kReadLocked); }
	__host__ void HostImage::SignalUnsetRead(cudaStream_t hostStream) { SignalChange << < 1, 1, 0, hostStream >> > (cu_deviceImage, DeviceImage::kReadLocked, DeviceImage::kUnlocked); }
	__host__ void HostImage::SignalSetWrite(cudaStream_t hostStream) { SignalChange << < 1, 1, 0, hostStream >> > (cu_deviceImage, DeviceImage::kUnlocked, DeviceImage::kWriteLocked); }
	__host__ void HostImage::SignalUnsetWrite(cudaStream_t hostStream) { SignalChange << < 1, 1, 0, hostStream >> > (cu_deviceImage, DeviceImage::kWriteLocked, DeviceImage::kUnlocked); }
	
	//void Image::Clear(Image* image, cudaStream_t hostStream) { Clear << < 1, 1, 0, hostStream >> > (image); }

	void HostImage::Create(unsigned int width, unsigned int height, cudaStream_t hostStream)
	{
		m_width = width;
		m_height = height;
		m_hostStream = hostStream;

		checkCudaErrors(cudaMalloc((void**)&cu_data, sizeof(float4) * width * height));

		checkCudaErrors(cudaMalloc((void**)&cu_deviceImage, sizeof(DeviceImage)));
		checkCudaErrors(cudaMemcpy(cu_deviceImage, this, sizeof(DeviceImage), cudaMemcpyHostToDevice));
	}

	__host__ void HostImage::Destroy()
	{
		if (!cu_deviceImage) { return; }
		//Assert(cu_deviceImage, "Trying to destroy an image that has already been destroyed.");

		checkCudaErrors(cudaFree((void*)cu_data));
		checkCudaErrors(cudaFree((void*)cu_deviceImage));

		std::printf("Destroyed image\n");
		cu_deviceImage = nullptr;
	}

	__global__ void KernelCopyImageToD3DTexture(unsigned int clientWidth, unsigned int clientHeight, DeviceImage* image, cudaSurfaceObject_t cuSurface)
	{
		if (*(image->AccessSignal()) != DeviceImage::kReadLocked) { return; }

		unsigned int kx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int ky = blockIdx.y * blockDim.y + threadIdx.y;

		if (kx >= clientWidth || ky >= clientHeight) { return; }

		int px = kx - clientWidth / 2 + image->Width() / 2;
		int py = ky - clientHeight / 2 + image->Height() / 2;

		if (px < 0 || px >= image->Width() || py < 0 || py >= image->Height()) { return; }

		surf2Dwrite(*(image->at(px, py)), cuSurface, kx * 16, ky);
	}

	// The host CPU Sinewave thread spawner
	__host__ void HostImage::CopyImageToD3DTexture(unsigned int clientWidth, unsigned int clientHeight, cudaSurfaceObject_t cuSurface, cudaStream_t hostStream)
	{
		dim3 block(16, 16, 1);
		dim3 grid(clientWidth / 16, clientHeight / 16, 1);

		SignalSetRead(hostStream);
		KernelCopyImageToD3DTexture << < grid, block, 0, hostStream >> > (clientWidth, clientHeight, cu_deviceImage, cuSurface);
		SignalUnsetRead(hostStream);

		getLastCudaError("CopyImageToD3DTexture execution failed.\n");
	}
}