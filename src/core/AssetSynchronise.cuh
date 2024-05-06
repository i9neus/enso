#pragma once

#include "Asset.cuh"

namespace Enso
{
	template<typename ObjectType, typename... ParameterPack>
	__global__ static void KernelSynchroniseTrivialParams(ObjectType* cu_object, ParameterPack... pack)
	{
		CudaAssert(cu_object);
		cu_object->Synchronise(pack...);
	}

	template<typename ObjectType, typename ParamsType, typename SuperType = ObjectType>
	__global__ static void KernelSynchroniseObjects(ObjectType* cu_object, const size_t hostParamsSize, const ParamsType* cu_params)
	{
		// Check that the size of the object in the device matches that of the host. Empty base optimisation can bite us here. 
		CudaAssert(cu_object);
		CudaAssert(cu_params);
		CudaAssert(sizeof(ParamsType) == hostParamsSize);

		// Force a validation check on the structure to make sure there aren't any dead pointers
		cu_params->Validate();

		// Push the data to the owning class
		cu_object->SuperType::Synchronise(*cu_params);
	}

	template<typename ObjectType>
	__global__ void KernelDestroyDeviceInstance(ObjectType* cu_instance)
	{
		if (cu_instance != nullptr) { delete cu_instance; }
	}

	template<typename ObjectType> __host__ void AssertIsTransferrableType()
	{
		//static_assert(std::is_standard_layout<ParamsType>::value, "Object must be standard layout type.");
		static_assert(!std::is_polymorphic<ObjectType>::value, "Object cannot be a polymorphic type.");
	}

	template<typename ObjectType>
	__host__ inline void AssertAllAreTrivialArguments(const ObjectType& t)
	{
		static_assert(std::is_trivial<ObjectType>::value, "Cannot use SynchroniseTrivialParams because at least one parameter type is non-trivial. Use SynchroniseObjects instead.");
	}

	template<typename ObjectType, typename... Pack>
	__host__ inline void AssertAllAreTrivialArguments(const ObjectType& t, const Pack... pack)
	{
		AssertAllAreTrivialArguments(t);
		AssertAllAreTrivialArguments(pack...);
	}

	template<typename ObjectType, typename... ParameterPack>
	__host__ void SynchroniseTrivialParams(ObjectType* cu_object, ParameterPack... pack)
	{
		Assert(cu_object);

		AssertAllAreTrivialArguments(pack...);

		KernelSynchroniseTrivialParams << <1, 1 >> > (cu_object, pack...);
		IsOk(cudaDeviceSynchronize());
	}

	template<typename ObjectType, typename ParamsType, typename SuperType = ObjectType>
	__host__ void SynchroniseObjects(ObjectType* cu_object, const ParamsType& params)
	{		
		AssertIsTransferrableType<ParamsType>();
		Assert(cu_object); 

		ParamsType* cu_params;
		IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
		IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

		IsOk(cudaDeviceSynchronize());
		KernelSynchroniseObjects <SuperType> << <1, 1 >> > (cu_object, sizeof(ParamsType), cu_params);
		IsOk(cudaDeviceSynchronize());

		IsOk(cudaFree(cu_params));
	}

	// FIXME: This needs to go away. Fix the Vector class first.
	template<typename ObjectType, typename ParamsType>
	__host__ void LegacySynchroniseObjects(ObjectType* cu_object, const ParamsType& params)
	{
		AssertIsTransferrableType<ParamsType>();
		Assert(cu_object);

		ParamsType* cu_params;
		IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
		IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

		IsOk(cudaDeviceSynchronize());
		KernelSynchroniseObjects <ObjectType> << <1, 1 >> > (cu_object, sizeof(ParamsType), cu_params);
		IsOk(cudaDeviceSynchronize());

		IsOk(cudaFree(cu_params));
	}
}