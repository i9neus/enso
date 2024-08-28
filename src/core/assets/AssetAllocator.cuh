#pragma once

#include "Asset.cuh"
#include "AssetSynchronise.cuh"

namespace Enso
{
	template<typename ObjectType, typename CastType>
	__global__ void KernelStaticCastOnDevice(ObjectType** inputPtr, CastType** outputPtr)
	{
		CudaAssert(inputPtr);
		CudaAssert(outputPtr);
		CudaAssert(*inputPtr);

		*outputPtr = static_cast<CastType*>(*inputPtr);
	}

	template<typename ObjectType, typename UpcastType, typename... Pack>
	__global__ void KernelCreateDeviceInstance(UpcastType** newInstance, Pack... args)
	{
		CudaAssert(newInstance);
		CudaAssert(!*newInstance);

		*newInstance = new ObjectType(args...);

		CudaAssert(*newInstance);
	}	

	enum MemoryAllocFlags : uint { kCudaMemoryManaged = 1 };

	class AssetAllocator
	{
	public:
		AssetAllocator() = delete;

		template<typename AssetType, typename... Args>
		__host__ static AssetHandle<AssetType> CreateAsset(const std::string& newId, Args... args)
		{
			static_assert(std::is_base_of<Host::Asset, AssetType>::value, "Asset type must be derived from Host::Asset");
			AssertMsg(!newId.empty(), "ID cannot be empty");

			auto& registry = GlobalResourceRegistry::Get();
			AssertMsgFmt(!registry.Exists(newId), "Object '%s' is already in asset registry!", newId.c_str());

			// Instantiate the new asset and set some handles
			//AssetHandle<AssetType> newAsset;
			//newAsset.m_ptr = std::make_shared<AssetType>(newId, args...);

			// Allocate memory for the new object
			// NOTE: For compilers other than msvc++, this function should be replaced with std::aligned_alloc()
			AssetType* naked = (AssetType*)_aligned_malloc(sizeof(AssetType), alignof(AssetType));
			Assert(naked);

			// Reset the shared pointer with a custom deleter for clean-up
			auto deleter = [](AssetType* ptr)
			{
				ptr->AssetType::~AssetType();
				_aligned_free(ptr);
			};
			AssetHandle<AssetType> newAsset;
			newAsset.m_ptr.reset(static_cast<AssetType*>(naked), deleter);

			// Create an initialisation context
			Host::Asset::InitCtx initCtx;
			initCtx.id = newId;
			initCtx.thisAssetHandle = newAsset.m_ptr;

			// Placement-new the object to construct it
			new (naked) AssetType(initCtx, args...);

			// Register the asset and return
			registry.RegisterAsset(newAsset.m_ptr, newId);
			return newAsset;
		}

		template<typename AssetType, typename... Args>
		__host__ static AssetHandle<AssetType> CreateChildAsset(const Host::Asset& parentAsset, const std::string& newId, Args... args)
		{
			static_assert(std::is_base_of<Host::Asset, AssetType>::value, "Asset type must be derived from Host::Asset");
			AssertMsg(!newId.empty(), "ID cannot be empty");

			// Concatenate new asset ID with its parent ID 
			const std::string& concatId = parentAsset.GetAssetID() + "/" + newId;

			auto& registry = GlobalResourceRegistry::Get();
			AssertMsgFmt(!registry.Exists(concatId), "Object '%s' is already in asset registry!", newId.c_str());

			// Instantiate the new asset and set some handles 
			// AssetHandle<AssetType> newAsset;
			//newAsset.m_ptr = std::make_shared<AssetType>(concatId, args...);

			// Allocate memory for the new object
			// NOTE: For compilers other than msvc++, this function should be replaced with std::aligned_alloc()
			AssetHandle<AssetType> newAsset;
			AssetType* naked = (AssetType*)_aligned_malloc(sizeof(AssetType), alignof(AssetType));
			Assert(naked);

			// Reset the shared pointer with a custom deleter for clean-up
			auto deleter = [](AssetType* ptr)
			{
				ptr->AssetType::~AssetType();
				_aligned_free(ptr);
			};
			newAsset.m_ptr.reset(static_cast<AssetType*>(naked), deleter);

			// Create an initialisation context
			Host::Asset::InitCtx initCtx;
			initCtx.id = concatId;
			initCtx.thisAssetHandle = newAsset.m_ptr;
			initCtx.parentAssetHandle = parentAsset.m_thisAssetHandle;

			// Placement-new the object to construct it
			new (naked) AssetType(initCtx, args...);

			registry.RegisterAsset(newAsset.m_ptr, concatId);
			return newAsset;
		}

		template<typename ObjectType>
		__host__ static void GuardedFreeDevice1DArray(const Host::Asset& parentAsset, const size_t numElements, ObjectType** deviceData) noexcept
		{
			Assert(deviceData);
			if (*deviceData != nullptr)
			{
				IsOk(cudaFree(*deviceData));
				*deviceData = nullptr;

				GlobalResourceRegistry::Get().DeregisterDeviceMemory(parentAsset.GetAssetID(), sizeof(ObjectType) * numElements);
			}
		}

		template<typename ObjectType>
		__host__ static void GuardedAllocDevice1DArray(const Host::Asset& parentAsset, const size_t numElements, ObjectType** deviceObject, const uint flags) noexcept
		{
			Assert(deviceObject);
			AssertMsg(*deviceObject == nullptr, "Memory is already allocated.");

			if (numElements > 0)
			{
				const size_t arraySize = sizeof(ObjectType) * numElements;
				if (flags & kCudaMemoryManaged)
				{
					IsOk(cudaMallocManaged((void**)deviceObject, arraySize));
				}
				else
				{
					IsOk(cudaMalloc((void**)deviceObject, arraySize));
				}

				GlobalResourceRegistry::Get().RegisterDeviceMemory(parentAsset.GetAssetID(), arraySize);
			}
		}

		template<typename ObjectType>
		__host__ static void GuardedAllocAndCopyToDevice1DArray(const Host::Asset& parentAsset, ObjectType** deviceObject, size_t numElements, ObjectType* hostData, const uint flags)
		{
			Assert(hostData);

			GuardedAllocDevice1DArray(parentAsset, numElements, deviceObject, flags);

			IsOk(cudaMemcpy(*deviceObject, hostData, sizeof(ObjectType) * numElements, cudaMemcpyHostToDevice));
		}

		template<typename Type>
		__host__ static void GuardedAllocDevice2DArray(const Host::Asset& parentAsset, const size_t width, const size_t height, cudaArray_t& deviceData)
		{			
			AssertMsgFmt(width > 0 && height > 0, "Invalid dimensions %i x %i", width, height);
			AssertMsg(deviceData == nullptr, "Memory is already allocated.");
			
			const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Type>();
			IsOk(cudaMallocArray(&deviceData, &channelDesc, width, height));

			GlobalResourceRegistry::Get().RegisterDeviceMemory(parentAsset.GetAssetID(), width * height * sizeof(Type));
		}

		template<typename Type>
		__host__ static void GuardedFreeDevice2DArray(const Host::Asset& parentAsset, const size_t width, const size_t height, cudaArray_t& deviceData) noexcept
		{
			if (deviceData != nullptr)
			{
				AssertMsgFmt(width > 0 && height > 0, "Invalid dimensions %i x %i", width, height);

				IsOk(cudaFreeArray(deviceData));
				deviceData = nullptr;

				GlobalResourceRegistry::Get().DeregisterDeviceMemory(parentAsset.GetAssetID(), width * height * sizeof(Type));
			}
		}

		// Frees a texture object of type cudaTextureObject_t
		__host__ static void GuardedFreeDeviceTextureObject(cudaTextureObject_t& obj) noexcept
		{
			if (obj)
			{
				IsOk(cudaDestroyTextureObject(obj));
				obj = 0;
			}
		}

		template<typename ObjectType, typename UpcastType = ObjectType, typename... Pack>
		__host__ static ObjectType* InstantiateOnDevice(const Host::Asset& parentAsset, Pack... args)
		{
			ObjectType** cu_tempBuffer;
			IsOk(cudaMalloc((void***)&cu_tempBuffer, sizeof(ObjectType*)));
			IsOk(cudaMemset(cu_tempBuffer, 0, sizeof(ObjectType*)));

			KernelCreateDeviceInstance<ObjectType, UpcastType> << <1, 1 >> > (cu_tempBuffer, args...);
			IsOk(cudaDeviceSynchronize());

			ObjectType* cu_data = nullptr;
			IsOk(cudaMemcpy(&cu_data, cu_tempBuffer, sizeof(ObjectType*), cudaMemcpyDeviceToHost));
			IsOk(cudaFree(cu_tempBuffer));

			GlobalResourceRegistry::Get().RegisterDeviceMemory(parentAsset.GetAssetID(), sizeof(ObjectType));

			Log::System("Instantiated device object at 0x%x\n", cu_data);
			return cu_data;
		}

		// Instantiate an instance of ObjectType, copies params to device memory, and passes it with the ctor parameters
		/*template<typename ObjectType, typename ParamsType, typename... Pack>
		__host__ static ObjectType* InstantiateOnDeviceWithParams(const Host::Asset& parentAsset, const ParamsType& params, Pack... args)
		{
			AssertIsTransferrableType<ParamsType>();

			ParamsType* cu_params;
			IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
			IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

			ObjectType* cu_data = AssetAllocator::InstantiateOnDevice<ObjectType>(parentAsset, cu_params, args...);

			IsOk(cudaFree(cu_params));
			return cu_data;
		}*/

		template<typename CastType, typename ObjectType>
		__host__ static CastType* StaticCastOnDevice(ObjectType* object)
		{
			static_assert(std::is_convertible<ObjectType*, CastType*>::value, "Can't statically cast between these inputs.");

			AssertMsg(object, "Device object is nullptr.");

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
		__host__ static void DestroyOnDevice(const Host::Asset& parentAsset, ObjectType*& cu_data) noexcept
		{
			if (cu_data == nullptr) { return; }

			KernelDestroyDeviceInstance << <1, 1 >> > (cu_data);
			IsOk(cudaDeviceSynchronize());

			GlobalResourceRegistry::Get().DeregisterDeviceMemory(parentAsset.GetAssetID(), sizeof(ObjectType));

			cu_data = nullptr;
		}
	};
}