#pragma once

#include "core/CudaHeaders.cuh"
#include "core/AssetSynchronise.cuh"
#include "core/HighResolutionTimer.h"

namespace Enso
{   
    struct RenderableObjectParams
    {
        __host__ __device__ RenderableObjectParams() :
            frameIdx(0),
            wallTime(0.) {}
        __device__ void Validate() const {}

        int     frameIdx;
        float   wallTime;
    };
    
    namespace Device
    {
        class RenderableObject
        {
        public:
            __device__ RenderableObject() {}
            __device__ void Synchronise(const RenderableObjectParams& params) { m_params = params; } 

        protected:
            RenderableObjectParams m_params;
        };
    }
    
    namespace Host
    {
        // Renderable objects are designed to be rapidly cycled by the inner loop
        class RenderableObject
        {
        public:
            __host__ RenderableObject() : cu_deviceInstance(nullptr) {}

            __host__ virtual void Render() = 0;
            __host__ bool Prepare()
            {
                m_params.frameIdx++;
                m_params.wallTime = m_wallTime.Get();

                SynchroniseObjects<Device::RenderableObject>(cu_deviceInstance, m_params);
                return true;
            }

        protected:
            __host__ void SetDeviceInstance(Device::RenderableObject* deviceInstance) { cu_deviceInstance = deviceInstance; }

        protected:
            RenderableObjectParams  m_params;
            HighResolutionTimer     m_wallTime;

        private:
            Device::RenderableObject* cu_deviceInstance;
        };
    }
}
