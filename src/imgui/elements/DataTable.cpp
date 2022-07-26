#include "IMGUIDataTable.h"

IMGUIDataTable::IMGUIDataTable(const std::string& id, const int numCols)
{
    m_numCols = numCols;
    m_cellIdx = -1;
    m_isEnded = false;

    Assert(ImGui::BeginTable(id.c_str(), m_numCols));
}

IMGUIDataTable::~IMGUIDataTable()
{
    AssertMsg(m_isEnded || std::uncaught_exception(), "IMGUIDataTable::End() was not called");
}

void IMGUIDataTable::End()
{
    ImGui::EndTable();
    m_isEnded = true;
}

void IMGUIDataTable::NextCell()
{
    ++m_cellIdx;
    if (m_cellIdx % m_numCols == 0)
    {
        ImGui::TableNextRow();
    }
    ImGui::TableSetColumnIndex(m_cellIdx % m_numCols);
}

IMGUIDataTable& IMGUIDataTable::operator <<(const std::string& str)
{
    NextCell();
    ImGui::Text(str.c_str());
    return *this;
}

IMGUIDataTable& IMGUIDataTable::operator <<(const float& f)
{
    NextCell();
    ImGui::Text(tfm::format("%f", f).c_str());
    return *this;
}

IMGUIDataTable& IMGUIDataTable::operator <<(const int& i)
{
    NextCell();
    ImGui::Text(tfm::format("%i", i).c_str());
    return *this;
}

IMGUIDataTable& IMGUIDataTable::operator <<(nullptr_t)
{
    NextCell();
    return *this;
}