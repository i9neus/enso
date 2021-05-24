#include "CppUnitTest.h"
#include "generic\StdIncludes.h"
#include "kernels\CudaMath.cuh"
#include "generic\StringUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Cuda;

namespace tests
{
	TEST_CLASS(CudaMatrixTests)
	{
	public:
		TEST_METHOD(TestMat3Transpose)
		{
			const mat3 matrix = { {0.6368585693091358f, 0.45275723409827173f, 0.5892859419083074f},
								  {0.6152364057023154f, 0.859458395158708f, 0.9169794155191895f},
								  {0.9239692167227462f, 0.8618463287946561f, 0.077336362022137f} };

			const mat3 invMatrixBaseline = { {0.6368585693091358f, 0.6152364057023154f, 0.9239692167227462f}, 
										     {0.45275723409827173f, 0.859458395158708f, 0.8618463287946561f}, 
											 {0.5892859419083074f, 0.9169794155191895f, 0.077336362022137f}};

			const mat3 invMatrixTest = transpose(matrix);

			constexpr float kEpsilon = 1e-6f;
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					Assert::IsTrue(std::abs(invMatrixBaseline[i][j] - invMatrixTest[i][j]) < kEpsilon, 
								   Widen(tfm::format("Elements are not equal: [%i, %i] %.10f should be %.10f", i, j, invMatrixBaseline[i][j], invMatrixTest[i][j])).c_str());
				}
			}
		}
		TEST_METHOD(TestMat3Determinant)
		{
			const mat3 matrix = { {0.4785538089186683f, 0.008696916817696687f, 0.34702929535492655f}, 
								  {0.1392798273842255f, 0.18060266455745766f, 0.5287012112413121f}, 
								  {0.5785867785984598f, 0.7603494187101738f, 0.40429785674104424f}};

			const float matrixDetBaseline = -0.1547758854005649f;

			const float matrixDetTest = det(matrix);

			constexpr float kEpsilon = 1e-6f;
			Assert::IsTrue(std::abs(matrixDetBaseline - matrixDetTest) < kEpsilon,
						   Widen(tfm::format("Elements are not equal: %.10f should be %.10f", matrixDetBaseline, matrixDetTest)).c_str());
		}

		TEST_METHOD(TestMat3Inverse)
		{
			const mat3 matrix = { {0.379200063703498f, 0.025939495319333483f, 0.8430596649677209f}, 
								  {0.22716932410774215f, 0.7994574121003679f, 0.07733502436821094f}, 
							      {0.6056191942042644f, 0.7646668584235827f, 0.20401107444364408} };

			const mat3 invMatrixBaseline = { {-0.4676699184577347f, -2.8761593503983978f, 3.0228824582301503f}, 
											 {-0.00220656355179829f, 1.9487796526551122f, -0.7296106723792858f}, 
											 {1.396576955275623f, 1.2337079941475406f, -1.337214357362365f} };

			const mat3 invMatrixTest = inverse(matrix);

			constexpr float kEpsilon = 1e-6f;
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					Assert::IsTrue(std::abs(invMatrixBaseline[i][j] - invMatrixTest[i][j]) < kEpsilon, Widen(tfm::format("Elements are not equal: [%i, %i] %.10f should be %.10f", i, j, invMatrixBaseline[i][j], invMatrixTest[i][j])).c_str());
				}
			}
		}
	};
}
