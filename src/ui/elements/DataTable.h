#pragma once

#include "WidgetUtils.h"

class IMGUIDataTable
{
public:
    IMGUIDataTable(const std::string& id, const int numCols);
    ~IMGUIDataTable();

    void End();

    IMGUIDataTable& operator <<(const std::string& str);
    IMGUIDataTable& operator <<(const int& i);
    IMGUIDataTable& operator <<(const float& f);
    IMGUIDataTable& operator <<(nullptr_t);

private:
    void NextCell();

    int m_numCols;
    int m_cellIdx;
    bool m_isEnded;
};
