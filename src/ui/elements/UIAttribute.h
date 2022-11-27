#pragma once

#include "core/math/Math.cuh"

namespace Enso
{
    namespace Json { class Node; }
    
    enum UIAttributeDataType : int
    {
        kUIDataUndefined = -1,

        kUIDataBool,
        kUIDataString,
        kUIDataInt,
        kUIDataInt2,
        kUIDataInt3,
        kUIDataFloat,
        kUIDataFloat2,
        kUIDataFloat3,
        kUIDataMat2,
        kUIDataMat3,
        kUIDataMat4,
    };

    enum UIAttributeWidgetType : int
    {
        kUIWidgetUndefined = -1,
        kUIWidgetDefault = 0,

        kUIWidgetString,
        kUIWidgetNumber,
        kUIWidgetSlider
    };

    class UIAttribute
    {
    public:
        UIAttribute();

        void Serialise(Json::Node&) const;
        void Deserialise(const Json::Node&);

        void Construct() const;
        bool IsDirty() const;
        void MakeClean();

    private:
        std::string     m_id;
        int             m_dataType;
        int             m_widgetType;
        
        union
        {
            int             m_intType;
            float           m_floatType;
            std::string     m_strType;
        };

        bool            m_isDirty;
    };
}
