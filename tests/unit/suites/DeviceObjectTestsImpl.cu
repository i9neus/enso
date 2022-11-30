#include "DeviceObjectTestsImpl.cuh"

#include "core/DeviceAllocator.cuh"

using namespace Enso;

namespace Tests
{	
	/*void DeviceObjectTestsImpl::ConstructDestruct()
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
	}*/	
}