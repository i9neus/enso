#pragma once

#include "CudaAsset.cuh"

namespace
{
	template<typename ObjectType, typename UpcastType, typename... Pack>
	__global__ void KernelCreateDeviceInstance(UpcastType** newInstance, Pack... args)
	{
		assert(newInstance);
		assert(!*newInstance);

		*newInstance = new ObjectType(args...);

		assert(*newInstance);
	}

	template<typename ObjectType, typename CastType>
	__global__ void KernelStaticCastOnDevice(ObjectType** inputPtr, CastType** outputPtr)
	{
		assert(inputPtr);
		assert(outputPtr);
		assert(*inputPtr);

		*outputPtr = static_cast<CastType*>(*inputPtr);
	}

	template<typename ObjectType, typename... ParameterPack>
	__global__ static void KernelSynchroniseTrivialParams(ObjectType* cu_object, ParameterPack... pack)
	{
		assert(cu_object);
		cu_object->Synchronise(pack...);
	}

	template<typename ObjectType, typename ParamsType>
	__global__ static void KernelSynchroniseObjects(ObjectType* cu_object, const size_t hostParamsSize, const ParamsType* cu_params)
	{
		// Check that the size of the object in the device matches that of the host. Empty base optimisation can bite us here. 
		assert(cu_object);
		assert(cu_params);
		assert(sizeof(ParamsType) == hostParamsSize);

		cu_object->Synchronise(*cu_params);
	}

	template<typename ObjectType>
	__global__ void KernelDestroyDeviceInstance(ObjectType* cu_instance)
	{
		if (cu_instance != nullptr) { delete cu_instance; }
	}
}

namespace Cuda
{
	enum MemoryAllocFlags : uint { kCudaMemoryManaged = 1 };
	
	namespace Host
    {
        class AssetAllocator : public Host::Asset
        {
        public:            
			AssetAllocator(const std::string& id) : Asset(id) {}

        protected:
			template<typename ObjectType>
			__host__ void GuardedFreeDeviceArray(const size_t numElements, ObjectType** deviceData) const
			{
				Assert(deviceData);
				if (*deviceData != nullptr)
				{
					IsOk(cudaFree(*deviceData));
					*deviceData = nullptr;

					GlobalResourceRegistry::Get().DeregisterDeviceMemory(GetAssetID(), sizeof(ObjectType) * numElements);
				}
			}

			template<typename ObjectType>
			__host__ inline void GuardedFreeDeviceObject(ObjectType** deviceData) const
			{
				GuardedFreeDeviceArray(1, deviceData);
			}

			template<typename ObjectType>
			void GuardedAllocDeviceArray(const size_t numElements, ObjectType** deviceObject, const uint flags = 0) const
			{
				Assert(deviceObject);
				AssertMsg(*deviceObject == nullptr, "Memory is already allocated.");

				if (numElements == 0) { return; }

				const size_t arraySize = sizeof(ObjectType) * numElements;

				if (flags & kCudaMemoryManaged)
				{
					IsOk(cudaMalloc((void**)deviceObject, arraySize));
				}
				else
				{
					IsOk(cudaMalloc((void**)deviceObject, arraySize));
				}

				GlobalResourceRegistry::Get().RegisterDeviceMemory(GetAssetID(), arraySize);
			}

			template<typename ObjectType>
			__host__ inline void GuardedAllocDeviceObject(ObjectType** deviceObject, const uint flags = 0) const
			{
				GuardedAllocDeviceArray(1, deviceObject, flags);
			}

			template<typename ObjectType>
			__host__ inline void GuardedAllocAndCopyToDeviceArray(ObjectType** deviceObject, size_t numElements, ObjectType* hostData, const uint flags = 0) const
			{
				Assert(hostData);

				GuardedAllocDeviceArray(numElements, deviceObject, flags);

				IsOk(cudaMemcpy(*deviceObject, hostData, sizeof(ObjectType) * numElements, cudaMemcpyHostToDevice));
			}
						
			template<typename ObjectType, typename UpcastType = ObjectType, typename... Pack>
			__host__ inline ObjectType* InstantiateOnDevice(Pack... args) const
			{
				ObjectType** cu_tempBuffer;
				IsOk(cudaMalloc((void***)&cu_tempBuffer, sizeof(ObjectType*)));
				IsOk(cudaMemset(cu_tempBuffer, 0, sizeof(ObjectType*)));

				KernelCreateDeviceInstance<ObjectType, UpcastType> << <1, 1 >> > (cu_tempBuffer, args...);
				IsOk(cudaDeviceSynchronize());

				ObjectType* cu_data = nullptr;
				IsOk(cudaMemcpy(&cu_data, cu_tempBuffer, sizeof(ObjectType*), cudaMemcpyDeviceToHost));
				IsOk(cudaFree(cu_tempBuffer));

				GlobalResourceRegistry::Get().RegisterDeviceMemory(GetAssetID(), sizeof(ObjectType));

				Log::System("Instantiated device object at 0x%x\n", cu_data);
				return cu_data;
			}

			// Instantiate an instance of ObjectType, copies params to device memory, and passes it with the ctor parameters
			template<typename ObjectType, typename ParamsType, typename... Pack>
			__host__ inline ObjectType* InstantiateOnDeviceWithParams(const ParamsType& params, Pack... args) const
			{
				static_assert(std::is_standard_layout<ParamsType>::value, "Parameter structure must be standard layout type.");

				ParamsType* cu_params;
				IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
				IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

				ObjectType* cu_data = InstantiateOnDevice<ObjectType>(cu_params, args...);

				IsOk(cudaFree(cu_params));
				return cu_data;
			}

			template<typename CastType, typename ObjectType>
			__host__ inline CastType* StaticCastOnDevice(ObjectType* object) const
			{
				static_assert(std::is_convertible<ObjectType*, CastType*>::value, "Can't statically cast between these inputs.");

				ObjectType** cu_inputPtr;
				IsOk(cudaMalloc((void***)&cu_inputPtr, sizeof(ObjectType*)));
				IsOk(cudaMemcpy(cu_inputPtr, &object, sizeof(ObjectType*), cudaMemcpyHostToDevice));
				CastType** cu_outputPtr;
				IsOk(cudaMalloc((void***)&cu_outputPtr, sizeof(CastType*)));
				IsOk(cudaMemset(cu_outputPtr, 0, sizeof(CastType*)));

				KernelStaticCastOnDevice << <1, 1 >> > (cu_inputPtr, cu_outputPtr);
				IsOk(cudaDeviceSynchronize());

				CastType* outputPtr = nullptr;
				IsOk(cudaMemcpy(&outputPtr, cu_outputPtr, sizeof(CastType*), cudaMemcpyDeviceToHost));
				IsOk(cudaFree(cu_inputPtr));
				IsOk(cudaFree(cu_outputPtr));

				return outputPtr;
			}

			template<typename ObjectType>
			__host__ inline void AreAllTrivialArguments(const ObjectType& t) const 
			{ 
				static_assert(std::is_trivial<ObjectType>::value, "Cannot sychronise because at least one parameter type is non-trivial.");
			}

			template<typename ObjectType, typename... Pack>
			__host__ inline void AreAllTrivialArguments(const ObjectType& t, const Pack... pack) const
			{
				AreAllTrivialArguments(t);
				AreAllTrivialArguments(pack...);
			}

			template<typename ObjectType, typename... ParameterPack>
			__host__ void SynchroniseTrivialParams(ObjectType* cu_object, ParameterPack... pack) const
			{
				Assert(cu_object);

				AreAllTrivialArguments(pack...);

				KernelSynchroniseTrivialParams << <1, 1 >> > (cu_object, pack...);
				IsOk(cudaDeviceSynchronize());
			}

			template<typename ObjectType, typename ParamsType>
			__host__ void SynchroniseObjects(ObjectType* cu_object, const ParamsType& params) const
			{
				Assert(cu_object);
				AssertMsg(std::is_standard_layout<ParamsType>::value, "Object structure must be standard layout type.");

				ParamsType* cu_params;
				IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
				IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

				IsOk(cudaDeviceSynchronize());
				KernelSynchroniseObjects << <1, 1 >> > (cu_object, sizeof(ParamsType), cu_params);
				IsOk(cudaDeviceSynchronize());

				IsOk(cudaFree(cu_params));
			}

			template<typename ObjectType>
			__host__ void DestroyOnDevice(ObjectType*& cu_data) const
			{
				if (cu_data == nullptr) { return; }

				KernelDestroyDeviceInstance << <1, 1 >> > (cu_data);
				IsOk(cudaDeviceSynchronize());

				GlobalResourceRegistry::Get().DeregisterDeviceMemory(GetAssetID(), sizeof(ObjectType));

				cu_data = nullptr;
			}
        };
    }
}