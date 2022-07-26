#pragma once

#include "thirdparty/imgui/imgui.h"
#include "generic/Math.h"

#include "kernels/cameras/CudaPerspectiveCamera.cuh"
#include "kernels/cameras/CudaLightProbeCamera.cuh"

#include "IMGUIAbstractShelf.h"
#include "generic/HighResolutionTimer.h"

// Perspective camera
class PerspectiveCameraShelf : public IMGUIShelf<Cuda::Host::PerspectiveCamera, Cuda::PerspectiveCameraParams>
{
public:
    PerspectiveCameraShelf(const Json::Node& json) : IMGUIShelf(json) {}
    virtual ~PerspectiveCameraShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new PerspectiveCameraShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final;
};

// Light probe camera
class LightProbeCameraShelf : public IMGUIShelf<Cuda::Host::LightProbeCamera, Cuda::LightProbeCameraParams>
{  
public:
    LightProbeCameraShelf(const Json::Node& json);
    virtual ~LightProbeCameraShelf() = default;

    static std::shared_ptr<IMGUIShelf> Instantiate(const Json::Node& json) { return std::shared_ptr<IMGUIShelf>(new LightProbeCameraShelf(json)); }
    virtual void Construct() override final;
    virtual void Jitter(const uint flags, const uint operation) override final {}
    virtual void OnUpdateRenderObjectStatistics(const Json::Node& shelfNode, const Json::Node& rootNode) override final;

    void Randomise();

private:
    std::vector<std::string>    m_swizzleLabels;

    struct HistogramWidgetData
    {
        std::vector<float>  data;
        float               maxValue;
    };

    struct ProbeGridStatistics
    {
        std::string                 gridID;
        float                       meanProbeValidity;
        float                       meanProbeDistance;
        int                         maxSamplesTaken;
        int                         minSamplesTaken;
        std::vector<HistogramWidgetData> histogramWidgetData;
        bool                        hasHistogram;
    };
    
    int                                 m_frameIdx;
    float                               m_bakeProgress;
    float                               m_bakeConvergence;
    float                               m_bakeProbesFull;
    float                               m_meanI, m_MSE;
    float                               m_meanFrameTime;    
    Cuda::vec2                          m_minMaxMeanI, m_minMaxMSE;
    Cuda::ivec2                         m_minMaxSamplesTaken;
    Cuda::vec2                          m_bakeEfficiency;
    float                               m_meanSamplesTaken;
    float                               m_meanProbeValidity;
    float                               m_meanProbeDistance;
    std::vector<float>                  m_meanIData, m_MSEData;
    std::vector<ProbeGridStatistics>    m_probeGridStatistics;

    HighResolutionTimer                 m_performanceTimer;
    std::vector<std::pair<float, float>> m_performanceLog;

private:
    void ExportStatsLog();
};