#pragma once

#include "../UIAttribute.h"

namespace Enso
{
    template<typename Type, int Dimension>
    class UIAttributeNumeric : public UIGenericAttribute
    {
    public:
        UIAttributeNumeric();
        virtual void Serialise(Json::Node&) const override final;
        virtual void Deserialise(const Json::Node&) override final;

    protected:
        virtual bool ConstructImpl() override final;

    private:
        std::array<Type, Dimension> m_data;
    };

    template class UIAttributeNumeric<float, 1>;
    template class UIAttributeNumeric<float, 2>;
    template class UIAttributeNumeric<float, 3>;
    template class UIAttributeNumeric<float, 4>;

    template class UIAttributeNumeric<int, 1>;
    template class UIAttributeNumeric<int, 2>;
    template class UIAttributeNumeric<int, 3>;
    template class UIAttributeNumeric<int, 4>;

    template class UIAttributeNumeric<bool, 1>;
}