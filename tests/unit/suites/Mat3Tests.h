#pragma once

#include "SuiteBase.h"

namespace Tests
{	
	TEST_CLASS(Mat3Tests), MatrixTestUtils
	{
	public:
		TEST_METHOD(TestMat3Trace)
		{
			const mat3 matrix = {{0.4356614267560044f, -0.9739092495469102f, 0.041087886064779866f},
				{-0.5821605178473233f, -0.4581920063276268f, 0.586103633723936f},
				{0.735760335795379f, 1.2810482561305214f, 0.21289357022313293f}};

			const float traceBaseline = 0.19036299065151052f;

			const float traceTest = trace(matrix);

			constexpr float kEpsilon = 1e-8f;
			Assert::IsTrue(std::abs(traceBaseline - traceTest) < kEpsilon,
				Widen(tfm::format("Elements are not equal: %.10f should be %.10f", traceTest, traceBaseline)).c_str());
		}
		TEST_METHOD(TestMat3Multiply)
		{
			const mat3 matrixA = { {0.3113761158140893f, 0.5660040004070195f, 0.6271553449468614f},
								   {0.1293134122192845f, 0.7916359845436349f, 0.6544086319924736f},
								   {0.10678317093971867f, 0.3618092618656048f, 0.8720116193879339f} };

			const mat3 matrixB = { {0.2942152063498533f, 0.7891131909779359f, 0.2150457540715247f},
								   {0.8977821251253189f, 0.1869049154486111f, 0.667945907017713f},
								   {0.3709102733463625f, 0.21709424438161395f, 0.31685194205855693f} };

			const mat3 matrixProdBaseline = { {0.8323782229063765f, 0.48765174590515026f, 0.6437355560714411f},
								   {0.9914894933617067f, 0.39207192358902265f, 0.7639289619128151f},
								   {0.6796811887888294f, 0.34119664186351006f, 0.5409308582018051f} };

			const mat3 matrixProdTest = matrixA * matrixB;

			TestMatIsEqual(matrixProdBaseline, matrixProdTest, 1e-6f);
		}

		TEST_METHOD(TestMat3Transpose)
		{
			const mat3 matrix = { {0.6368585693091358f, 0.45275723409827173f, 0.5892859419083074f},
								  {0.6152364057023154f, 0.859458395158708f, 0.9169794155191895f},
								  {0.9239692167227462f, 0.8618463287946561f, 0.077336362022137f} };

			const mat3 transMatrixBaseline = { {0.6368585693091358f, 0.6152364057023154f, 0.9239692167227462f},
											 {0.45275723409827173f, 0.859458395158708f, 0.8618463287946561f},
											 {0.5892859419083074f, 0.9169794155191895f, 0.077336362022137f}};

			const mat3 transMatrixTest = transpose(matrix);

			TestMatIsEqual(transMatrixBaseline, transMatrixTest, 1e-6f);
		}
		TEST_METHOD(TestMat3Determinant)
		{
			// Find the determinant of a 3x3 matrix

			const mat3 matrix = { {-0.042892382162663445f, -0.9826061663646066f, -0.3059414092901469f},
								  {-0.721440345231549f, -0.6387946708850847f,  0.05740242248262417f},
								  {0.1571735571969195f, 0.5206988374203476f, -0.19140428651791153f} };

			const float matrixDetBaseline = 0.2070681993539183f;

			const float matrixDetTest = det(matrix);

			constexpr float kEpsilon = 1e-6f;
			Assert::IsTrue(std::abs(matrixDetBaseline - matrixDetTest) < kEpsilon,
						   Widen(tfm::format("Elements are not equal: %.10f should be %.10f", matrixDetTest, matrixDetBaseline)).c_str());
		}

		TEST_METHOD(TestMat3Inverse)
		{
			// Inverse of a non-singular 3x3 matrix

			const mat3 matrix = { {0.6347789803421424f, -0.7771607777375271f, 0.5790519892677031f},
								  {-0.6243937065879472f, -0.5172780650846991f, -0.8685224809824379f},
								  {0.08449324101924827f, -0.5376909865279451f, -0.20798783690282585f}};

			const mat3 invMatrixBaseline = { {-2.40364128057734f, -3.163249205240259f, 6.517303140808326f},
											 {-1.3592905338495636f, -1.210164984782027f, 1.2690915550313424f},
											 {2.5375850527577772f, 1.8434810070449061f, -5.441236574213772f}};

			const mat3 invMatrixTest = inverse(matrix);

			TestMatIsEqual(invMatrixBaseline, invMatrixTest, 1e-5f);
		}

		TEST_METHOD(TestMat3SingularInverse)
		{
			// Inverse of a singular matrix. Since this has no inverse, the function should just return null (a zero matrix)

			const mat3 matrix = { {1.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 1.0f} };

			const mat3 invMatrixTest = inverse(matrix);

			TestMatIsEqual(mat3::Null(), invMatrixTest, 1e-5f);
		}

		/*TEST_METHOD(TestMat3RotationalSymmetry)
		{
			mat3 rotXMat = RotXMat3(0.765190f);
			Assert::IsTrue(rotXMat.IsSymmetric(), Widen(tfm::format("rotXMat is not symmetric: %s", rotXMat.format())).c_str());

			mat3 rotYMat = RotYMat3(1.265190f);
			Assert::IsTrue(rotYMat.IsSymmetric(), Widen(tfm::format("rotYMat is not symmetric: %s", rotYMat.format())).c_str());

			mat3 rotZMat = RotZMat3(-0.965190f);
			Assert::IsTrue(rotZMat.IsSymmetric(), Widen(tfm::format("rotZMat is not symmetric: %s", rotZMat.format())).c_str());

			mat3 productMat = mat3::Indentity();
			productMat *= rotXMat;
			productMat *= rotYMat;
			productMat *= rotZMat;
			Assert::IsTrue(productMat.IsSymmetric(), Widen(tfm::format("productMat is not symmetric: %s", productMat.format())).c_str());

			mat3 transMat = transpose(productMat);
			Assert::IsTrue(transMat == productMat, Widen(tfm::format("transMat and productMat are not equal: %s -> %s", productMat.format(), transMat.format())).c_str());
		}

		TEST_METHOD(TestMat3CompoundTransform)
		{
			BidirectionalTransform transform = CreateCompoundTransform(vec3(0.765190f, 1.265190f, -0.965190f));

			Assert::IsTrue(transform.fwd.IsSymmetric(), Widen(tfm::format("Forward matrix is not symmetric: %s", transform.fwd.format())).c_str());
		}*/
	};
}