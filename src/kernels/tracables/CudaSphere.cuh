﻿#pragma once

#include "CudaTracable.cuh"

namespace Cuda
{
    class RenderObjectContainer;
    namespace Host {  class Sphere;   }
    namespace Json {  class Node;  }

    namespace Device
    {
        class Sphere : public Device::Tracable
        {
            friend Host::Sphere;
        protected:
            BidirectionalTransform m_transform;

        public:
            __device__ Sphere() = default;
            __device__ ~Sphere() = default;
            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const override final; 
            __device__ void Synchronise(const BidirectionalTransform& transform)
            {
                m_transform = transform;
            }
        };
    }

    namespace Host
    {        
        class Sphere : public Host::Tracable
        {
        private:
            Device::Sphere* cu_deviceData;
            Device::Sphere  m_hostData;

        public:
            __host__ Sphere(const ::Json::Node& json);
            __host__ virtual ~Sphere() = default;
            __host__ virtual void OnDestroyAsset() override final;
            
            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual Device::Sphere* GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ static std::string GetAssetTypeString() { return "sphere"; }
            
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;
        };
    }
}