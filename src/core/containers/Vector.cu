#include "Vector.cuh"

namespace Enso
{
	void test()
	{
		std::vector<int>::iterator;
	}
	
	/*template<typename HostType, typename DeviceType>
	__host__ void Host::VectorBase<HostType, DeviceType>::DestructiveResizeImpl(const size_t newSize)
	{
		// Deallocate the old device memory
		AssetAllocator::GuardedFreeDevice1DArray(*this, m_deviceData.capacity, &m_deviceData.mem);

		if (newSize == 0)
		{
			m_deviceData.mem = nullptr;
			m_deviceData.capacity = 0;
			m_deviceData.size = 0;
		}
		else
		{
			AssetAllocator::GuardedAllocDevice1DArray(*this, newSize, &m_deviceData.mem, 0u);
			m_deviceData.capacity = newSize;
			m_deviceData.size = newSize;
		}
	}

	// Zeroes the contents of the vector
	template<typename HostType, typename DeviceType>
	__host__ void Host::VectorBase<HostType, DeviceType>::Wipe()
	{
		IsOk(cudaMemset(cu_deviceData, 0, sizeof(DeviceType) * m_deviceData.size));
		std::memset(m_hostData.data(), 0, sizeof(HostType) * m_hostData.size());
	}

	template<typename CommonType>
	__host__ void Host::Vector<CommonType>::Synchronise(const uint syncFlags)
	{
		if (syncFlags == kVectorSyncUpload)
		{
			// Make sure the device data matches the host size
			if (m_hostData.size() != m_deviceData.size)
			{
				DestructiveResizeImpl(m_hostData.size());
			}

			if (!m_hostData.empty())
			{
				// Upload the data and synchronise with the device object
				IsOk(cudaMemcpy(m_deviceData.mem, m_hostData.data(), sizeof(CommonType*) * m_hostData.size(), cudaMemcpyHostToDevice));
				SynchroniseObjects<Device::Vector<CommonType>>(cu_deviceInstance, m_deviceData);
			}			
		}
		else
		{
			if (m_deviceData.size > 0)
			{
				// We're assuming that the host content is already initialised
				Assert(m_deviceData.mem);
				Assert(m_hostData.size() == m_deviceData.size);
				IsOk(cudaMemcpy(m_hostData.data(), m_deviceData.mem, sizeof(CommonType) * m_deviceData.size, cudaMemcpyDeviceToHost));
			}
		}
	}
	
	template<typename HostType, typename DeviceType>
	__host__ void Host::AssetVector<HostType, DeviceType>::Synchronise(const uint syncFlags)
	{			
		AssertMsg((syncFlags | kVectorSyncUpload) == 0, "Using kVectorSyncDownload on AssetVector type is meaningless.");
				
		// Make sure the device data matches the host size
		if (m_hostData.size() != m_deviceData.size)
		{
			DestructiveResizeImpl(m_hostData.size());
		}

		if(!m_hostData.empty())
		{
			std::vector<DeviceType*> devicePtrs(m_hostData.size());
			for (int idx = 0; idx < m_hostData.size(); ++idx)
			{
				devicePtrs[idx] = m_hostData[idx]->GetDeviceInstance();
			}

			IsOk(cudaMemcpy(m_deviceData.mem, devicePtrs.data(), sizeof(DeviceType*) * devicePtrs.size(), cudaMemcpyHostToDevice));
			SynchroniseObjects<Device::Vector<DeviceType>>(cu_deviceInstance, m_deviceData);
		}		
	}*/

} // namespace Enso