#include "DeviceObjectTestsImpl.cuh"

#include <kernels/DeviceAllocator.cuh>

using namespace Cuda;

class NonAssetClass
{
public:
	NonAssetClass() = default;
};

enum EchoValue : int { kBaseClass = 1, kSuperClassA, kSuperClassB, kTopClass };

class BaseClass : public Device::Asset
{
public:
	__host__ __device__ BaseClass() : m_id(kBaseClass) {}
	__device__ virtual int Virtual() { return m_id; };
	__device__ int NonVirtual() { return m_id; };
private:
	int m_id;
};

class SuperClassA : virtual public BaseClass
{
public:
	__host__ __device__ SuperClassA() : m_id(kSuperClassA) {}
	__device__ virtual int Virtual() override { return m_id; }
	__device__ int NonVirtual() { return m_id; };
private:
	int m_id;
};

class SuperClassB : virtual public BaseClass
{
public:
	__host__ __device__ SuperClassB() : m_id(kSuperClassB) {}
	__device__ virtual int Virtual() override { return m_id;  }
	__device__ int NonVirtual() { return m_id; };
private:
	int m_id;
};

class TopClass : public SuperClassA, 
				 public SuperClassB
{
public:
	__host__ __device__ TopClass() : m_id(kTopClass) {}
	__device__ virtual int Virtual() override { return m_id; }
	__device__ int NonVirtual() { return m_id; };
private:
	int m_id;
};

template<typename Type>
__global__ void KernelEchoObject(Type* object, int* result)
{
	result[0] = object->Virtual();
	result[1] = object->NonVirtual();
}

template<typename BaseType>
void VerifyObject(BaseType* devicePtr, const int id)
{
	int* deviceResult;
	int hostResult[2];
	IsOk(cudaMalloc(&deviceResult, sizeof(int) * 2));
	KernelEchoObject << <1, 1 >> > (devicePtr, deviceResult);
	IsOk(cudaMemcpy(&hostResult, deviceResult, sizeof(int) * 2, cudaMemcpyDeviceToHost));
	IsOk(cudaFree(deviceResult));
	IsOk(cudaDeviceSynchronize());

	Assert::IsTrue(hostResult[0] == kTopClass,
		Widen(tfm::format("Virtual call from object cast from %s to %s returned %i instead of %i.",
			typeid(TopClass).name(), typeid(BaseType).name(), hostResult[0], kTopClass)).c_str());

	Assert::IsTrue(hostResult[1] == id,
		Widen(tfm::format("Non-virtual call from object cast from %s to %s returned %i instead of %i.",
			typeid(TopClass).name(), typeid(BaseType).name(), hostResult[1], id)).c_str());
}

namespace Tests
{	
	void DeviceObjectTestsImpl::ConstructDestruct()
	{
		// Try to instantiate a device object that's not derived from Device::Asset
		//DeviceObject<NonAssetClass> nonAssetInstance = CreateDeviceObject<NonAssetClass>();

		Log::EnableLevel(kLogSystem, true);

		DeviceObject<TopClass> deviceObj = CreateDeviceObject<TopClass>();
		deviceObj.DestroyObject();
	}

	void DeviceObjectTestsImpl::Cast()
	{
		Log::EnableLevel(kLogSystem, true);		

		DeviceObject<TopClass> deviceObj = CreateDeviceObject<TopClass>();
		
		// Cast the top class pointer to the base class pointers
		TopClass* topPtr = static_cast<TopClass*>(deviceObj);
		SuperClassA* superAPtr = static_cast<SuperClassA*>(deviceObj);
		SuperClassB* superBPtr = static_cast<SuperClassB*>(deviceObj);
		BaseClass* basePtr = static_cast<BaseClass*>(deviceObj);
		basePtr = static_cast<BaseClass*>(deviceObj);

		// Verify that the objects are working correctly
		VerifyObject(topPtr, kTopClass);
		VerifyObject(superAPtr, kSuperClassA);
		VerifyObject(superBPtr, kSuperClassB);
		VerifyObject(basePtr, kBaseClass);

		deviceObj.DestroyObject();
	}
	
}