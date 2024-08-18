#pragma once

#include "core/assets/GenericObject.cuh"
#include "core/containers/Vector.cuh"
#include "core/math/Quarternion.cuh"

namespace Enso
{
    struct GaussianPoint
    {
        vec3        p;
        Quaternion  rot;
        vec3        sca;
        vec4        rgba;
    };

    struct GaussianPointCloudObjects
    {
        __device__ void Validate() const
        { 
            CudaAssert(splats != nullptr);
        }

        Device::Vector<GaussianPoint>* splats = nullptr;
    };
    
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class GaussianPointCloud : public Device::GenericObject
        {
        public:
            __device__ GaussianPointCloud() {}
            __device__ void Synchronise(const GaussianPointCloudObjects& objects) { m_objects = objects; }

        private:
            GaussianPointCloudObjects m_objects;
        };
    }

    namespace Host
    {
        class GaussianPointCloud : public Host::GenericObject
        {
        public:
            __host__ GaussianPointCloud(const Asset::InitCtx& initCtx);
            __host__ virtual ~GaussianPointCloud();
            __host__ Device::GaussianPointCloud* GetDeviceInstance() { return cu_deviceInstance; }
            __host__ virtual void Synchronise(const uint syncFlags) override final;
            __host__ void AppendSplats(const std::vector<GaussianPoint>& points);
            __host__ void Finalise(); 
            __host__ Host::Vector<GaussianPoint>& GetSplatList() { return *m_hostSplatList; }

            __host__ static const std::string  GetAssetClassStatic() { return "gaussianpointcloud"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

        protected:
            Device::GaussianPointCloud* cu_deviceInstance = nullptr;
            GaussianPointCloudObjects   m_deviceObjects;

            AssetHandle<Host::Vector<GaussianPoint>> m_hostSplatList;
        };
    }
}