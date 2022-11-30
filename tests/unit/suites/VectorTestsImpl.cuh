#include "SuiteBase.h"

namespace Tests
{
    class VectorTestsImpl : public SuiteBase
    {
    public:
        VectorTestsImpl() = default;

        void ConstructDestruct();
        void Resize();
        void EmplaceBack();
        void Synchronise();

    private:
       
    };
}
