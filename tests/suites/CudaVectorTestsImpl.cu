#include "CudaVectorTestsImpl.cuh"

#include <kernels/CudaVector.cuh>

using namespace Cuda;

namespace Tests
{
	template<typename T>
	inline Host::Vector<int> ConstructImpl(const uint size, const uint flags)
	{
		return Host::Vector<int>(tfm::format("cudaVector_%i_%i", size, flags), size, flags, nullptr);
	}

	void CudaVectorTestsImpl::ConstructDestruct()
	{
		const std::vector<uint> sizeList = { 0, 1, 10, 100, 1000 };
		const std::vector<uint> flagsList = { 0u, kVectorHostAlloc, kVectorHostAlloc | kVectorUnifiedMemory };

		// Construct for different permutations of flags
		for (const uint flags : flagsList)
		{
			// Construct for different sizes
			for (const uint size : sizeList)
			{	
				auto vec = ConstructImpl<int>(size, flags);
			}
		}
	}

	// Create a set of arrays and resize them a certain number of times
	void CudaVectorTestsImpl::Resize()
	{
		constexpr int kNumIterations = 10;
		constexpr int kNumResizes = 10;
		constexpr int kMaxArraySize = 1000;

		// Make sure the resizing operation behaves the way it's supposed to
		for (int iterIdx = 0; iterIdx < kNumIterations; ++iterIdx)
		{
			const uint startSize = RandInt(0, kMaxArraySize);
			auto vec = ConstructImpl<int>(startSize, kVectorHostAlloc);

			// Initialise with some random data
			std::vector<int> reference(startSize);
			for (int idx = 0; idx < reference.size(); ++idx)
			{
				reference[idx] = RandInt(-1e5, 1e5);
				vec[idx] = reference[idx];
			}

			uint lastCapacity = 0u;
			for (int resizeIdx = 0; resizeIdx < kNumResizes; ++resizeIdx)
			{
				const uint oldSize = vec.Size();
				const uint newSize = RandInt(0, kMaxArraySize);
				
				vec.Resize(newSize);
				reference.resize(newSize);

				// If the array has grown, add new data
				if (newSize > oldSize)
				{
					for (int idx = oldSize; idx < newSize; ++idx)
					{
						reference[idx] = RandInt(-1e5, 1e5);
						vec[idx] = reference[idx];
					}
				}

				// Check to make sure the vector is reporting the correct size and that its capacity hasn't shrunk
				Assert::IsTrue(vec.Size() == newSize, Widen(tfm::format("Size changed from %i. Expected %i", vec.Size(), newSize)).c_str());
				Assert::IsTrue(vec.Capacity() >= lastCapacity, Widen(tfm::format("Capacity changed from %i to %i", lastCapacity, vec.Capacity())).c_str());

				// Check to make sure that the data are properly preserved by the resizing operation
				for (int idx = 0; idx < std::min(newSize, oldSize); ++idx)
				{
					Assert::IsTrue(vec[idx] == reference[idx], Widen(tfm::format("Data not preserved during resize. Element at %i is %i. Expected %i.", idx, reference[idx], vec[idx])).c_str());
				}				
			}
		}
	}	

	void CudaVectorTestsImpl::EmplaceBack()
	{

	}

	template<typename ElementType, typename HostType, typename DeviceType>
	__global__ void KernelSquareValues(Device::VectorBase<ElementType, HostType, DeviceType>* cu_vector)
	{
		if (kKernelIdx < cu_vector->Size())
		{
			(*cu_vector)[kKernelIdx] *= (*cu_vector)[kKernelIdx];
		}
	}

	void CudaVectorTestsImpl::Synchronise()
	{
		const size_t kRefSize = 10;
		std::vector<int> referenceVec(kRefSize);
		for (auto& i : referenceVec) { i = RandInt(-1e5, 1e5); }

		std::string inputListStr;
		Host::Vector<int> hostVec(tfm::format("cudaVector"), 0u, kVectorHostAlloc, nullptr);
		for (const auto i : referenceVec)
		{
			hostVec.PushBack(i);
			inputListStr += tfm::format("%i ", i);
		}
		Logger::WriteMessage(inputListStr.c_str());

		Assert::IsTrue(hostVec.Size() == kRefSize, Widen(tfm::format("Vector size is %i. Expected %i.", hostVec.Size(), kRefSize)).c_str());

		// Upload the data back to the device
		hostVec.Synchronise(kVectorSyncUpload);

		// Modify the values on the device
		const int blockSize = 16 * 16;
		const int gridSize = (hostVec.Size() + (blockSize - 1)) / blockSize;				
		KernelSquareValues << < gridSize, blockSize >> > (hostVec.GetDeviceInstance());

		// Download the data back to the host
		hostVec.Synchronise(kVectorSyncDownload);

		Assert::IsTrue(hostVec.Size() == kRefSize, Widen(tfm::format("Vector size is %i. Expected %i.", hostVec.Size(), kRefSize)).c_str());

		std::string outputListStr;
		for (int idx = 0; idx < hostVec.Size(); ++idx)
		{
			const int expected = sqr(referenceVec[idx]);
			const int actual = hostVec[idx];

			Assert::IsTrue(expected == actual, Widen(tfm::format("Value at index %i is %i. Expected %i", idx, actual, expected)).c_str());
			outputListStr += tfm::format("%i ", actual);
		}
		Logger::WriteMessage(outputListStr.c_str());
	}
}