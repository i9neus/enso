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

	namespace Host
    {
		enum MemoryAllocFlags : uint { kCudaMemoryManaged = 1 };
		
		class AssetAllocator
        {
		private:
			const Asset & const m_parentAsset;

        public:            
			AssetAllocator(const Asset& parent) : m_parentAsset(parent) {}

			template<typename AssetType, typename... Args>
			__host__ static AssetHandle<AssetType> CreateAsset(const std::string& newId, Args... args)
			{
				static_assert(std::is_base_of<Host::Asset, AssetType>::value, "Asset type must be derived from Host::Asset");

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
				Asset::InitCtx initCtx;
				initCtx.id = newId;
				initCtx.thisAssetHandle = newAsset.m_ptr;

				// Placement-new the object to construct it
				new (naked) AssetType(initCtx, args...);

				// Register the asset and return
				registry.RegisterAsset(newAsset.m_ptr, newId);
				return newAsset;
			}

			template<typename AssetType, typename... Args>
			__host__ inline AssetHandle<AssetType> CreateChildAsset(const std::string& newId, Args... args) const
			{
				static_assert(std::is_base_of<Host::Asset, AssetType>::value, "Asset type must be derived from Host::Asset");

				// Concatenate new asset ID with its parent ID 
				const std::string& concatId = m_parentAsset.GetAssetID() + "/" + newId;

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
				Asset::InitCtx initCtx;
				initCtx.id = newId;
				initCtx.thisAssetHandle = newAsset.m_ptr;
				initCtx.parentAssetHandle = m_parentAsset.m_thisAssetHandle;

				// Placement-new the object to construct it
				new (naked) AssetType(initCtx, args...);

				registry.RegisterAsset(newAsset.m_ptr, concatId);
				return newAsset;
			}

			template<typename ObjectType>
			__host__ void GuardedFreeDeviceArray(const size_t numElements, ObjectType** deviceData) const
			{
				Assert(deviceData);
				if (*deviceData != nullptr)
				{
					IsOk(cudaFree(*deviceData));
					*deviceData = nullptr;

					GlobalResourceRegistry::Get().DeregisterDeviceMemory(m_parentAsset.GetAssetID(), sizeof(ObjectType) * numElements);
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

				GlobalResourceRegistry::Get().RegisterDeviceMemory(m_parentAsset.GetAssetID(), arraySize);
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

				GlobalResourceRegistry::Get().RegisterDeviceMemory(m_parentAsset.GetAssetID(), sizeof(ObjectType));

				Log::System("Instantiated device object at 0x%x\n", cu_data);
				return cu_data;
			}

			// Instantiate an instance of ObjectType, copies params to device memory, and passes it with the ctor parameters
			template<typename ObjectType, typename ParamsType, typename... Pack>
			__host__ inline ObjectType* InstantiateOnDeviceWithParams(const ParamsType& params, Pack... args) const
			{
				AssertIsTransferrableType<ParamsType>();

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

				AssertMsgFmt(object, "Device object belonging to '%s' is nullptr.", m_parentAsset.GetAssetID());

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
			__host__ void DestroyOnDevice(ObjectType*& cu_data) const
			{
				if (cu_data == nullptr) { return; }

				KernelDestroyDeviceInstance << <1, 1 >> > (cu_data);
				IsOk(cudaDeviceSynchronize());

				GlobalResourceRegistry::Get().DeregisterDeviceMemory(m_parentAsset.GetAssetID(), sizeof(ObjectType));

				cu_data = nullptr;
			}
        };		
    }
}