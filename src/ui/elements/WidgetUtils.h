#pragma once

#include "thirdparty/imgui/imgui.h"
#include "io/json/JsonUtils.h"

#include "core/math/Math.cuh"
#include "core/Asset.cuh"

namespace Enso
{
    namespace Json { class Document; class Node; }

    class UIStyle
    {
    public:
        UIStyle::UIStyle(const int shelfIdx)
        {
            /*const float alpha = 0.8f * shelfIdx++ / float(::max(1ull, m_shelves.size() - 1));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(alpha, 0.5f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(alpha, 0.6f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(alpha, 0.7f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(alpha, 0.9f, 0.9f));
            ImGui::PushStyleColor(ImGuiCol_Header, (ImVec4)ImColor::HSV(alpha, 0.9f, 0.7f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, (ImVec4)ImColor::HSV(alpha, 0.9f, 0.8f));*/

            const float lightness = mix(0.5f, 0.35f, float(shelfIdx % 2));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.5f * lightness));
            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.5f * lightness));
            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.5f * lightness));
            ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.9f * lightness));
            ImGui::PushStyleColor(ImGuiCol_Header, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.6f * lightness));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.6f * lightness));
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.6f * lightness));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.0f, 0.0f, 0.9f * lightness));
            ImGui::PushStyleColor(ImGuiCol_CheckMark, (ImVec4)ImColor::HSV(0.0f, 0.0f, 1.0f));
        }

        UIStyle::~UIStyle()
        {
            ImGui::PopStyleColor(9);
        }
    };

    // Little RAII class to make sure indents are closed when going out of scope
    class ImGuiIndent
    {
    public:
        ImGuiIndent(const float margin = -1.0f) : m_margin(margin)
        {
            if (m_margin >= 0.0f) { ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, m_margin); }
            ImGui::Indent();
        }

        ~ImGuiIndent()
        {
            ImGui::Unindent();
            if (m_margin >= 0.0f) { ImGui::PopStyleVar(); }
        }

    private:
        float m_margin;
    };

#define SL ImGui::SameLine()

    static void ToolTip(const char* desc)
    {
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    static void HelpMarker(const char* desc)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }
}