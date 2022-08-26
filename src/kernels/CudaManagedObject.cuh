#pragma once

#include "AssetAllocator.cuh"

namespace Cuda
{
	/*
		Simple wrapper class that allows us to manage device objects using the Asset interface. As with all objects,
		template type T must be a standard layout type to use this object. 
	*/
	namespace Host
	{
		template<typename T, typename = typename std::enable_if<std::is_standard_layout<T>::value>::type>
		class ManagedObject : public Host::AssetAllocator
		{
		private:
			T*	cu_deviceData;

		public:
			__host__ ManagedObject(const std::string& id) : AssetAllocator(id), cu_deviceData(nullptr) { GuardedAllocDeviceObject(&cu_deviceData); }
			__host__ ManagedObject(const std::string& id, const T& hostObject) : ManagedObject(id) {	SynchroniseDevice(hostObject); }
			__host__  ~ManagedObject() { OnDestroyAsset(); }

			__host__ void SynchroniseDevice(const T& object) const { Assert(cu_deviceData);  IsOk(cudaMemcpy((void*)cu_deviceData, (void*)&hostData, sizeof(T), cudaMemcpyHostToDevice)); }
			__host__ void SynchroniseHost(T& hostObject) { Assert(cu_deviceData); IsOk(cudaMemcpy((void*)&hostObject, (void*)cu_deviceData, sizeof(T), cudaMemcpyDeviceToHost)); }

			__host__  virtual void OnDestroyAsset() override final	{ GuardedFreeDeviceObject(&cu_deviceData); }
			__host__ T* GetDeviceInstance() const { Assert(cu_deviceData); return cu_deviceData; }
		};	
	}
}