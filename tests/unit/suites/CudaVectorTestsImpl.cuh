#include "SuiteBase.h"

namespace Tests
{
    class CudaVectorTestsImpl : public SuiteBase
    {
    public:
        CudaVectorTestsImpl() = default;

        void ConstructDestruct();
        void Resize();
        void EmplaceBack();
        void Synchronise();

    private:
       
    };
}
