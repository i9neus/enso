#pragma once
#pragma once

#include "SuiteBase.h"

namespace Tests
{
	TEST_CLASS(CudaMat2Tests), MatrixTestUtils
	{
	public:
		TEST_METHOD(TestMat2Trace)
		{
			const mat2 matrix = {{0.4356614267560044, -0.9739092495469102}, {0.041087886064779866, -0.5821605178473233}};

			const float traceBaseline = -0.14649909109131887;

			const float traceTest = trace(matrix);

			constexpr float kEpsilon = 1e-8f;
			Assert::IsTrue(std::abs(traceBaseline - traceTest) < kEpsilon,
				Widen(tfm::format("Elements are not equal: %.10f should be %.10f", traceTest, traceBaseline)).c_str());
		}
		TEST_METHOD(TestMat2Multiply)
		{
			const mat2 matrixA = {{0.3113761158140893, 0.5660040004070195}, {0.6271553449468614,  0.1293134122192845}};

			const mat2 matrixB = {{0.7916359845436349, 0.6544086319924736}, {0.10678317093971867, 0.3618092618656048}};

			const mat2 matrixProdBaseline = { {0.30693623993388686, 0.4085527075852714}, {0.5102872351606218, 0.45720266155782546}};

			const mat2 matrixProdTest = matrixA * matrixB;

			TestMatIsEqual(matrixProdBaseline, matrixProdTest, 1e-6f);
		}

		TEST_METHOD(TestMat2Transpose)
		{
			const mat2 matrix = {{0.44447961821288695, - 0.7811027183432904}, {-0.05859455523361223,0.07116365077185227}};

			const mat2 transMatrixBaseline = { {0.44447961821288695, -0.05859455523361223}, {-0.7811027183432904,  0.07116365077185227} };

			const mat2 transMatrixTest = transpose(matrix);

			TestMatIsEqual(transMatrixBaseline, transMatrixTest, 1e-6f);
		}
		TEST_METHOD(TestMat2Determinant)
		{
			// Find the determinant of a 3x3 matrix

			const mat2 matrix = {{-0.042892382162663445, -0.9826061663646066}, {-0.3059414092901469, -0.721440345231549}};

			const float matrixDetBaseline = -0.26967562031954084;

			const float matrixDetTest = det(matrix);

			constexpr float kEpsilon = 1e-6f;
			Assert::IsTrue(std::abs(matrixDetBaseline - matrixDetTest) < kEpsilon,
						   Widen(tfm::format("Elements are not equal: %.10f should be %.10f", matrixDetTest, matrixDetBaseline)).c_str());
		}

		TEST_METHOD(TestMat2Inverse)
		{
			// Inverse of a non-singular 3x3 matrix

			const mat2 matrix = {{0.6347789803421424, - 0.7771607777375271}, {0.5790519892677031, - 0.6243937065879472}};

			const mat2 invMatrixBaseline = { {-11.635136397721492, 14.481843036740804}, {-10.790225470590512,  11.82865865040089}};

			const mat2 invMatrixTest = inverse(matrix);

			TestMatIsEqual(invMatrixBaseline, invMatrixTest, 1e-5f);
		}

		TEST_METHOD(TestMat2SingularInverse)
		{
			// Inverse of a singular matrix. Since this has no inverse, the function should just return null (a zero matrix)

			const mat2 matrix = { {1.0f, 1.0f}, {0.0f, 0.0f} };

			const mat2 invMatrixTest = inverse(matrix);

			TestMatIsEqual(mat2::Null(), invMatrixTest, 1e-5f);
		}

		/*TEST_METHOD(TestMat2RotationalSymmetry)
		{
			mat2 rotXMat = RotXMat2(0.765190f);
			Assert::IsTrue(rotXMat.IsSymmetric(), Widen(tfm::format("rotXMat is not symmetric: %s", rotXMat.format())).c_str());

			mat2 rotYMat = RotYMat2(1.265190f);
			Assert::IsTrue(rotYMat.IsSymmetric(), Widen(tfm::format("rotYMat is not symmetric: %s", rotYMat.format())).c_str());

			mat2 rotZMat = RotZMat2(-0.965190f);
			Assert::IsTrue(rotZMat.IsSymmetric(), Widen(tfm::format("rotZMat is not symmetric: %s", rotZMat.format())).c_str());

			mat2 productMat = mat2::Indentity();
			productMat *= rotXMat;
			productMat *= rotYMat;
			productMat *= rotZMat;
			Assert::IsTrue(productMat.IsSymmetric(), Widen(tfm::format("productMat is not symmetric: %s", productMat.format())).c_str());

			mat2 transMat = transpose(productMat);
			Assert::IsTrue(transMat == productMat, Widen(tfm::format("transMat and productMat are not equal: %s -> %s", productMat.format(), transMat.format())).c_str());
		}

		TEST_METHOD(TestMat2CompoundTransform)
		{
			BidirectionalTransform transform = CreateCompoundTransform(vec3(0.765190f, 1.265190f, -0.965190f));

			Assert::IsTrue(transform.fwd.IsSymmetric(), Widen(tfm::format("Forward matrix is not symmetric: %s", transform.fwd.format())).c_str());
		}*/
	};
}