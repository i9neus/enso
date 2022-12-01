#include "UIAttributeNumeric.h"

#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

#include "io/json/JsonUtils.h"
#include "io/Serialisable.cuh"
#include "core/math/Math.cuh"

namespace Enso
{
    // Min/max for n-tuples
#define FIND_MINMAX_N(Type, Op) \
	Type minmaxVal = v[0]; \
	for (int idx = 1; idx < size; ++idx) { minmaxVal = Op(minmaxVal, v[idx]); } \
	return minmaxVal;

    template<typename Type> __host__ __device__ Type maxn(const Type* v, const int size) { FIND_MINMAX_N(Type, max); }
    template<typename Type> __host__ __device__ Type minn(const Type* v, const int size) { FIND_MINMAX_N(Type, min); }
    template<> __host__ __device__ float maxn(const float* v, const int size) { FIND_MINMAX_N(float, maxf); }
    template<> __host__ __device__ float minn(const float* v, const int size) { FIND_MINMAX_N(float, minf); }
    
    template<typename Type, int Dimension>
    UIAttributeNumeric<Type, Dimension>::UIAttributeNumeric()
    {
        for (auto& el : m_data) { el = Type(0); }
    }
    
    // Drag
    template<typename Type, int Dim> inline bool ConstructDrag(std::array<Type, Dim>& data, const char* label, const vec2& dataRange) 
    { 
        ImGui::Text("Unsupported component.");
        return false;  
    }   
    template<> inline bool ConstructDrag(std::array<float, 1>& data, const char* label, const vec2& dataRange) 
    { return ImGui::DragFloat(label, data.data(), maxf(0.00001f, data[0] * 0.01f), dataRange[0], dataRange[1], "%.6f"); }
    template<> inline bool ConstructDrag(std::array<float, 2>& data, const char* label, const vec2& dataRange) 
    { return ImGui::DragFloat2(label, data.data(), maxf(0.00001f, maxf(data[0], data[1]) * 0.01f), dataRange[0], dataRange[1], "%.6f"); }
    template<> inline bool ConstructDrag(std::array<float, 3>& data, const char* label, const vec2& dataRange) 
    { return ImGui::DragFloat3(label, data.data(), maxf(0.00001f, max3(data[0], data[1], data[2])* 0.01f), dataRange[0], dataRange[1], "%.6f"); }
    template<> inline bool ConstructDrag(std::array<float, 4>& data, const char* label, const vec2& dataRange) 
    { return ImGui::DragFloat4(label, data.data(), maxf(0.00001f, maxn(data.data(), 4) * 0.01f), dataRange[0], dataRange[1], "%.6f"); }
    template<> inline bool ConstructDrag(std::array<int, 1>& data, const char* label, const vec2& dataRange) 
    { return ImGui::DragInt(label, data.data(), int(maxf(0.00001f, data[0] * 0.01f)), int(dataRange[0]), int(dataRange[1]), "%i"); }
    template<> inline bool ConstructDrag(std::array<int, 2>& data, const char* label, const vec2& dataRange) 
    { return ImGui::DragInt2(label, data.data(), maxf(0.00001f, max(data[0], data[1]) * 0.01f), int(dataRange[0]), int(dataRange[1]), "%i"); }
    template<> inline bool ConstructDrag(std::array<int, 3>& data, const char* label, const vec2& dataRange) 
    { return ImGui::DragInt3(label, data.data(), maxf(0.00001f, max3(data[0], data[1], data[2])* 0.01f), int(dataRange[0]), int(dataRange[1]), "%i"); }
    template<> inline bool ConstructDrag(std::array<int, 4>& data, const char* label, const vec2& dataRange) 
    { return ImGui::DragInt4(label, data.data(), maxf(0.00001f, maxn(data.data(), 4) * 0.01f), int(dataRange[0]), int(dataRange[1]), "%i"); }

    // Slider
    template<typename Type, int Dim> inline bool ConstructSlider(std::array<Type, Dim>& data, const char* label, const vec2& dataRange) 
    { 
        ImGui::Text("[Unsupported component]"); return false;  
    }   
    template<> inline bool ConstructSlider<float, 1>(std::array<float, 1>& data, const char* label, const vec2& dataRange) 
    { return ImGui::SliderFloat(label, data.data(), dataRange[0], dataRange[1], "%.6f"); }
    template<> inline bool ConstructSlider<float, 2>(std::array<float, 2>& data, const char* label, const vec2& dataRange) 
    { return ImGui::SliderFloat2(label, data.data(), dataRange[0], dataRange[1], "%.6f"); }
    template<> inline bool ConstructSlider<float, 3>(std::array<float, 3>& data, const char* label, const vec2& dataRange) 
    { return ImGui::SliderFloat3(label, data.data(), dataRange[0], dataRange[1], "%.6f"); }
    template<> inline bool ConstructSlider<float, 4>(std::array<float, 4>& data, const char* label, const vec2& dataRange) 
    { return ImGui::SliderFloat4(label, data.data(), dataRange[0], dataRange[1], "%.6f"); }
    template<> inline bool ConstructSlider<int, 1>(std::array<int, 1>& data, const char* label, const vec2& dataRange) 
    { return ImGui::SliderInt(label, data.data(), int(dataRange[0]), int(dataRange[1]), "%i"); }
    template<> inline bool ConstructSlider<int, 2>(std::array<int, 2>& data, const char* label, const vec2& dataRange) 
    { return ImGui::SliderInt2(label, data.data(), int(dataRange[0]), int(dataRange[1]), "%i"); }
    template<> inline bool ConstructSlider<int, 3>(std::array<int, 3>& data, const char* label, const vec2& dataRange) 
    { return ImGui::SliderInt3(label, data.data(), int(dataRange[0]), int(dataRange[1]), "%i"); }
    template<> inline bool ConstructSlider<int, 4>(std::array<int, 4>& data, const char* label, const vec2& dataRange) 
    { return ImGui::SliderInt4(label, data.data(), int(dataRange[0]), int(dataRange[1]), "%i"); }

    // Input box
    template<typename Type, int Dim> inline bool ConstructInput(std::array<Type, Dim>& data, const char* label, const vec2& dataRange) 
    { 
        ImGui::Text("[Unsupported component]");  return false;
    }   
    template<> inline bool ConstructInput<bool, 1>(std::array<bool, 1>& data, const char* label, const vec2& dataRange)
    {
        return ImGui::Checkbox(label, data.data());
    }
    template<> inline bool ConstructInput<float, 1>(std::array<float, 1>& data, const char* label, const vec2& dataRange) 
    { return ImGui::InputFloat(label, data.data()); }
    template<> inline bool ConstructInput<float, 2>(std::array<float, 2>& data, const char* label, const vec2& dataRange) 
    { return ImGui::InputFloat2(label, data.data()); }
    template<> inline bool ConstructInput<float, 3>(std::array<float, 3>& data, const char* label, const vec2& dataRange) 
    { return ImGui::InputFloat3(label, data.data()); }
    template<> inline bool ConstructInput<float, 4>(std::array<float, 4>& data, const char* label, const vec2& dataRange) 
    { return ImGui::InputFloat4(label, data.data()); }
    template<> inline bool ConstructInput<int, 1>(std::array<int, 1>& data, const char* label, const vec2& dataRange) 
    { return ImGui::InputInt(label, data.data()); }
    template<> inline bool ConstructInput<int, 2>(std::array<int, 2>& data, const char* label, const vec2& dataRange) 
    { return ImGui::InputInt2(label, data.data()); }
    template<> inline bool ConstructInput<int, 3>(std::array<int, 3>& data, const char* label, const vec2& dataRange) 
    { return ImGui::InputInt3(label, data.data()); }
    template<> inline bool ConstructInput<int, 4>(std::array<int, 4>& data, const char* label, const vec2& dataRange) 
    { return ImGui::InputInt4(label, data.data()); }

    // Colour picker
    template<typename Type, int Dim> inline bool ConstructColourPicker(std::array<Type, Dim>& data, const char* label)
    {
        ImGui::Text("[Colour pickers only supported for float3 types.]"); return false;
    }
    template<> inline bool ConstructColourPicker(std::array<float, 3>& data, const char* label)
    {
        return ImGui::ColorEdit3(label, data.data(), ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_Float);
    }

    template<typename Type, int Dimension>
    bool UIAttributeNumeric<Type, Dimension>::ConstructImpl()
    {
        switch (m_uiWidget.type)  
        {  
        case kUIWidgetDrag:  
            return ConstructDrag(m_data, m_uiWidget.label.c_str(), m_dataRange);
        case kUIWidgetSlider:  
            return ConstructSlider(m_data, m_uiWidget.label.c_str(), m_dataRange);
        case kUIWidgetColourPicker:
            return ConstructColourPicker(m_data, m_uiWidget.label.c_str());
        default:
            return ConstructInput(m_data, m_uiWidget.label.c_str(), m_dataRange);
        }
    }

    template<typename Type, int Dim> void SerialiseImpl(const std::array<Type, Dim>& data, const std::string& id, Json::Node& node)
    {
        std::vector<Type> vec(data.begin(), data.end());
        node.AddArray(id, vec, Json::kPathIsDAG);
    }

    template<typename Type> void SerialiseImpl(const std::array<Type, 1>& data, const std::string& id, Json::Node& node)
    {
        node.AddValue(id, data[0], Json::kPathIsDAG);
    }

    template<typename Type, int Dimension>
    void UIAttributeNumeric<Type, Dimension>::Serialise(Json::Node& node) const
    {
        SerialiseImpl(m_data, m_id, node);
    }

    template<typename Type, int Dim> bool DeserialiseImpl(std::array<Type, Dim>& data, const std::string& id, const Json::Node& node)
    {
        std::vector<Type> vec(data.begin(), data.end());
        bool success = node.GetArrayValues(id, vec, Json::kRequiredWarn | Json::kPathIsDAG);
        AssertMsgFmt(vec.size() == Dim, "'%s' expected %i elements, found %i", id.c_str(), Dim, vec.size());
        std::copy(vec.begin(), vec.end(), data.begin());
        return success;
    }

    template<typename Type> bool DeserialiseImpl(std::array<Type, 1>& data, const std::string& id, const Json::Node& node)
    {
        return node.GetValue(id, data[0], Json::kRequiredWarn | Json::kPathIsDAG);
    }

    template<typename Type, int Dimension>
    void UIAttributeNumeric<Type, Dimension>::Deserialise(const Json::Node& node)
    {
        DeserialiseImpl(m_data, m_id, node);
    }



}