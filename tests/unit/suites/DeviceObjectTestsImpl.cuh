#include "SuiteBase.h"

namespace Tests
{
    class DeviceObjectTestsImpl : public SuiteBase
    {
    public:
        DeviceObjectTestsImpl() = default;

        void ConstructDestruct();
        void Cast();

    private:

    };
}
