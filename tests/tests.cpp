#include "CppUnitTest.h"
#include "generic\StdIncludes.h"
#include "kernels\math\CudaMath.cuh"
#include "kernels\CudaRay.cuh"
#include "generic\StringUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Cuda;

namespace tests
{
	class MatrixTestUtils
	{
	public:
		template<typename T>
		void TestMatIsEqual(const T& matBaseline, const T& matTest, const float kEpsilon)
		{
			for (int i = 0; i < T::kDims; i++)
			{
				for (int j = 0; j < T::kDims; j++)
				{
					Assert::IsTrue(std::abs(matBaseline[i][j] - matTest[i][j]) < kEpsilon,
						Widen(tfm::format("Elements are not equal: [%i, %i] %.10f should be %.10f", i, j, matTest[i][j], matBaseline[i][j])).c_str());
				}
			}
		}

		template<typename T>
		void TestVecIsEqual(const T& vecBaseline, const T& vecTest, const float kEpsilon, const char* name)
		{
			for (int i = 0; i < T::kDims; i++)
			{
				Assert::IsTrue(std::abs(vecBaseline[i] - vecTest[i]) < kEpsilon,
						Widen(tfm::format("Element %i of %s is not equal: %s should be %s", i, name, vecTest.format(), vecBaseline.format())).c_str());
			}
		}

		template<typename T>
		void CheckTypeSize(size_t targetSize, const char* name)
		{
			Assert::IsTrue(sizeof(T) == targetSize,
				Widen(tfm::format("%s is the wrong size: %i should be %i", name, sizeof(T), targetSize)).c_str());
		}
	}; 

	TEST_CLASS(CudaMathTests), MatrixTestUtils
	{
	public:
		TEST_METHOD(TestBidirTransformScale)
		{
			// Test that the matrix decomposition is correct for BidirectionalTransform object.

			const vec3 scale(0.183276f, 2.083f, 6.0385f);
			const vec3 rotate(1.5124f, -0.873651f, 0.58367f);
			const vec3 translate(0.0f);

			const BidirectionalTransform transform = CreateCompoundTransform(rotate, translate, scale);

			TestVecIsEqual(scale, transform.scale, 1e-7f, "scale");
		}

		/*TEST_METHOD(TestBidirObjectToWorld)
		{
			// Transform a hit point from object space into world space

			const BidirectionalTransform transform = CreateCompoundTransform(vec3(kHalfPi, 0.0f, 0.0f));

			const vec3 oObject(0.0f, 1.0f, 0.0f);
			const vec3 nObject(0.0f, 0.0f, 1.0f);
			const vec3 oWorld(0.0f, 0.0f, -1.0f);
			const vec3 nWorld(0.0f, 1.0f, 0.0f);

			HitPoint hitWorld = transform.HitToWorldSpace(HitPoint(oObject, nObject));

			TestVecIsEqual(hitWorld.o, oWorld, 1e-7f, "origin");
		}

		TEST_METHOD(TestBidirWorldToObjectToWorld)
		{
			// Transform a random vector and normal into object space and back out again

			const BidirectionalTransform transform = CreateCompoundTransform(vec3(-kHalfPi, 0.0f, 0.0f));

			const vec3 oWorld(10.3765f, 8.987291f, -2.579872f);
			const vec3 nWorld = normalize(vec3(-1.826f, 0.825176f, -0.127982376f));

			const HitPoint hitObject = ToObjectSpace( HitPoint(oWorld, nWorld));
			const HitPoint hitWorld = transform.HitToWorldSpace(hitObject);

			TestVecIsEqual(hitWorld.o, oWorld, 1e-6f, "origin");
			TestVecIsEqual(hitWorld.n, nWorld, 1e-6f, "normal");
		}*/
	};
	
	TEST_CLASS(CudaVecTests), MatrixTestUtils
	{
	public:
		TEST_METHOD(TestMathStructSizes)
		{
			CheckTypeSize<vec2>(sizeof(float) * 2, "vec2");
			CheckTypeSize<vec3>(sizeof(float) * 3, "vec3");
			CheckTypeSize<vec4>(sizeof(float) * 4, "vec4");
			CheckTypeSize<ivec2>(sizeof(int) * 2, "ivec2");
			CheckTypeSize<ivec3>(sizeof(int) * 3, "ivec3");
			CheckTypeSize<ivec4>(sizeof(int) * 4, "ivec4");
			CheckTypeSize<uvec2>(sizeof(unsigned int) * 2, "uvec2");
			CheckTypeSize<uvec3>(sizeof(unsigned int) * 3, "uvec3");
			CheckTypeSize<uvec4>(sizeof(unsigned int) * 4, "uvec4");
		}
		TEST_METHOD(TestMathOrthoBasis)
		{
			// Test the CreateBasis(n) function by checking that three orthogonal vectors, when transformed, are still orthogonal

			const vec3 b0 = normalize(vec3(0.73571f, -1.2945f, 0.34517f));
			const vec3 b1 = normalize(cross(b0, (abs(dot(b0, vec3(1.0, 0.0, 0.0))) < 0.5) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
			const vec3 b2 = cross(b1, b0);	

			constexpr float kDotEpsilon = 1e-6f;
			Assert::IsTrue(std::abs(dot(b0, b1)) < kDotEpsilon && std::abs(dot(b1, b2)) < kDotEpsilon && std::abs(dot(b2, b0)) < kDotEpsilon,
				L"Test basis is not orthnormal.");

			{
				const vec3 n = normalize(vec3(-0.537851f, -0.872652f, 1.275876f));

				const mat3 onbTest = CreateBasis(n);
				const vec3 t0 = onbTest * b0;
				const vec3 t1 = onbTest * b1;
				const vec3 t2 = onbTest * b2;

				Assert::IsTrue(std::abs(dot(t0, t1)) < kDotEpsilon, Widen(tfm::format("t0 and t1 are not ortogonal: dot product = %f", dot(t0, t1))).c_str());
				Assert::IsTrue(std::abs(dot(t1, t2)) < kDotEpsilon, Widen(tfm::format("t1 and t2 are not ortogonal: dot product = %f", dot(t1, t2))).c_str());
				Assert::IsTrue(std::abs(dot(t2, t0)) < kDotEpsilon, Widen(tfm::format("t2 and t0 are not ortogonal: dot product = %f", dot(t2, t0))).c_str());
			}

			// Test the CreateBasis(n, up) function by checking that three orthogonal vectors, when transformed, are still orthogonal

			{
				const vec3 n = normalize(vec3(-0.537851f, -0.872652f, 1.275876f));
				const vec3 up = normalize(vec3(1.283508f, 0.7673658f, -0.38762f));

				const mat3 onbTest = CreateBasis(n, up);
				const vec3 s0 = onbTest * b0;
				const vec3 s1 = onbTest * b1;
				const vec3 s2 = onbTest * b2;

				Assert::IsTrue(std::abs(dot(s0, s1)) < kDotEpsilon, Widen(tfm::format("s0 and s1 are not ortogonal: dot product = %f", dot(s0, s1))).c_str());
				Assert::IsTrue(std::abs(dot(s1, s2)) < kDotEpsilon, Widen(tfm::format("s1 and s2 are not ortogonal: dot product = %f", dot(s1, s2))).c_str());
				Assert::IsTrue(std::abs(dot(s2, s0)) < kDotEpsilon, Widen(tfm::format("s2 and s0 are not ortogonal: dot product = %f", dot(s2, s0))).c_str());
			}
		}

		TEST_METHOD(TestMathVecCast)
		{
			{
				ivec4 a(1, 2, 3, 4);
				vec4 b(a);
				ivec4 c(b);
				Assert::IsTrue(a.x == c.x && a.y == c.y && a.z == c.z && a.w == c.w,
					Widen(tfm::format("a and c are not the same: %s should be %s", a.format(), c.format())).c_str());
			}

			{
				ivec3 a(1, 2, 3);
				vec3 b(a);
				ivec3 c(b);
				Assert::IsTrue(a.x == c.x && a.y == c.y && a.z == c.z,
					Widen(tfm::format("a and c are not the same: %s should be %s", a.format(), c.format())).c_str());
			}

			{
				ivec2 a(1, 2);
				vec2 b(a);
				ivec2 c(b);
				Assert::IsTrue(a.x == c.x && a.y == c.y,
					Widen(tfm::format("a and c are not the same: %s should be %s", a.format(), c.format())).c_str());
			}
		}
		TEST_METHOD(TestVec3Swizzle)
		{
			const vec3 a = {1.0f, 2.0f, 3.0f};
			
			{
				const vec3 b = a.zyx;
				const vec3 c = { 3.0f, 2.0f, 1.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z,
					Widen(tfm::format("zyx: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}

			{
				const vec3 b = a.yzx;
				const vec3 c = { 2.0f, 3.0f, 1.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z,
					Widen(tfm::format("yzx: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}

			{
				const vec3 b = a.xxx;
				const vec3 c = { 1.0f, 1.0f, 1.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z,
					Widen(tfm::format("xxx: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}

			{
				const vec3 b = a.yyz;
				const vec3 c = { 2.0f, 2.0f, 3.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z,
					Widen(tfm::format("yyz: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}
		}

		TEST_METHOD(TestVec4Swizzle)
		{
			const vec4 a = { 1.0f, 2.0f, 3.0f, 4.0f };

			{
				const vec4 b = a.wzyx;
				const vec4 c = { 4.0f, 3.0f, 2.0f, 1.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z && b.w == c.w,
					Widen(tfm::format("wzyx: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}	

			{
				const vec4 b = a.yxwz;
				const vec4 c = { 2.0f, 1.0f, 4.0f, 3.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z && b.w == c.w,
					Widen(tfm::format("yxwz: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}

			{
				const vec4 b = a.zzyx;
				const vec4 c = { 3.0f, 3.0f, 2.0f, 1.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z && b.w == c.w,
					Widen(tfm::format("zzyx: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}

			{
				const vec4 b = a.wwww;
				const vec4 c = { 4.0f, 4.0f, 4.0f, 4.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z && b.w == c.w,
					Widen(tfm::format("wwww: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}

			{
				vec4 b(1.0f);
				b.xy -= vec2(0.5f);
				const vec4 c = { 0.5f, 0.5f, 1.0f, 1.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z && b.w == c.w,
					Widen(tfm::format("b.xy -= vec2(0.5f): b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}
		}

		TEST_METHOD(TestVecSwizzleCast)
		{
			const vec4 a = { 1.0f, 2.0f, 3.0f, 4.0f };

			{
				const vec3 b = a.wzy;
				const vec3 c = { 4.0f, 3.0f, 2.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y && b.z == c.z,
					Widen(tfm::format("wzy: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}

			{
				const vec2 b = a.yz;
				const vec2 c = { 2.0f, 3.0f };

				Assert::IsTrue(b.x == c.x && b.y == c.y,
					Widen(tfm::format("yz: b and c are not the same: %s should be %s", b.format(), c.format())).c_str());
			}
		}		
		TEST_METHOD(TestVecSwizzleArithmetic)
		{
			vec4 a = { 2.0f, 3.0f, 5.0f, 7.0f };
			const vec4 b = { 11.0f, 13.0f, 17.0f, 19.0f };
			const vec4 c = { a.x + b.y + b.w, a.y + b.x, a.z, a.w + b.z };

			a.yxwx += b;

			Assert::IsTrue(a.x == c.x && a.y == c.y && a.z == c.z && a.w == c.w,
					Widen(tfm::format("a.yxwx += b.wyzz: a and c are not the same: %s should be %s", a.format(), c.format())).c_str());
		}
		TEST_METHOD(TestVec3Arithmetic)
		{
			const vec3 a = {1.4521684705132136f, -0.6657411666062907f, 1.3685779839015542f };
			const vec3 b = {-0.2759170976270484f, -0.8027837214736566f, 0.626739861528872f };
			const vec3 c = {0.188018244645761f, 1.1014213458267346f, -0.3645220628376178f };
			const vec3 d = { 0.26855194801984617f, -0.2575156574079758f, 1.9315152852223294f };
			const float e = 1.775825603536588f;
			const float f = 0.7341684559831378;

			const vec3 arithBaseline = { 3.0399700612483183f, -2.5112281028785945f, -17.784401459345407f };
			vec3 r;

			r = a + b;
			r = r - c;
			r = -r;
			r = r * d;
			r = r * e;
			r += a;
			r -= b;
			r *= e;
			r /= f;

			constexpr float kEpsilon = 1e-6f;
			Assert::IsTrue(cwiseMax(abs(arithBaseline - r)) < kEpsilon,
				Widen(tfm::format("Elements are not equal: %s should be %s", r.format(), arithBaseline.format())).c_str());
		}		

		TEST_METHOD(TestVec3Products)
		{
			const vec3 a0 = {-0.2575156574079758, 1.9315152852223294, 1.475488818465541};
			const vec3 b0 = {1.775825603536588, 0.7341684559831378, - 0.12139078960623273};
			const float dotBaseline = 0.7816439441941685f;
			const float dotTest = dot(a0, b0);

			constexpr float kEpsilon = 1e-7f;
			Assert::IsTrue(abs(dotBaseline - dotTest) < kEpsilon,
				Widen(tfm::format("Elements are not equal: %.10f should be %.10f", dotBaseline, dotTest)).c_str());

			const vec3 a1 = {1.775825603536588f, 0.7341684559831378f, - 0.12139078960623273f};
			const vec3 b1 = {-0.37584680565822204f, 0.7414234438824758f, - 0.6135374960846884f};
			const vec3 crossBaseline = {-0.36043789890279004f, 1.1351599347867507f, 1.5925736037050646f};
			const vec3 crossTest = cross(a1, b1);

			Assert::IsTrue(cwiseMax(abs(crossBaseline - crossTest)) < kEpsilon,
				Widen(tfm::format("Elements are not equal: %s should be %s", crossTest.format(), crossBaseline.format())).c_str());
		}

		TEST_METHOD(TestVec4Arithmetic)
		{
			const vec4 a = {1.4521684705132136f, -0.6657411666062907f, 1.3685779839015542f, -0.4365905598819211f };
			const vec4 b = {-0.2759170976270484f, - 0.8027837214736566f, 0.626739861528872f, - 0.30653647979191767f };
			const vec4 c = {0.188018244645761f, 1.1014213458267346f, - 0.3645220628376178f, 1.2459706444488443f };
			const vec4 d = { 0.26855194801984617f, -0.2575156574079758f, 1.9315152852223294f, 1.475488818465541f };
			const float e = 1.775825603536588f;
			const float f = 0.7341684559831378;

			const vec4 arithBaseline = { 3.0399700612483183f, -2.5112281028785945f, -17.784401459345407f, 12.291991850318022f };
			vec4 r;

			r = a + b;
			//Logger::WriteMessage(tfm::format("%s\n", r.format()).c_str());
			r = r - c;
			//Logger::WriteMessage(tfm::format("%s\n", r.format()).c_str());
			r = -r;
			//Logger::WriteMessage(tfm::format("%s\n", r.format()).c_str());
			r = r * d;
			//Logger::WriteMessage(tfm::format("%s\n", r.format()).c_str());
			r = r * e;
			//Logger::WriteMessage(tfm::format("%s\n", r.format()).c_str());
			r += a;
			//Logger::WriteMessage(tfm::format("%s\n", r.format()).c_str());
			r -= b;
			//Logger::WriteMessage(tfm::format("%s\n", r.format()).c_str());
			r *= e;
			//Logger::WriteMessage(tfm::format("%s\n", r.format()).c_str());
			r /= f;
			//Logger::WriteMessage(tfm::format("%s\n", r.format()).c_str());

			constexpr float kEpsilon = 1e-6f;
			Assert::IsTrue(cwiseMax(abs(arithBaseline - r)) < kEpsilon,
				Widen(tfm::format("Elements are not equal: %s should be %s", r.format(), arithBaseline.format())).c_str());
		}
	};

	TEST_CLASS(CudaMat4Tests), MatrixTestUtils
	{
	public:
		TEST_METHOD(TestMat4Trace)
		{
			const mat4 matrix = {{0.4356614267560044, - 0.9739092495469102, 0.041087886064779866, - 0.5821605178473233}, 
				{-0.4581920063276268, 0.586103633723936, 0.735760335795379, 1.2810482561305214}, 
				{0.21289357022313293, 1.711178841763223, 1.927712744248094, 0.8610800567938739}, 
				{-0.7059955441512078, 1.9452641621621032, 1.2505180113330017, - 0.5988375302772724}};

			const float traceBaseline = 2.350640274450762f;

			const float traceTest = trace(matrix);

			constexpr float kEpsilon = 1e-6f;
			Assert::IsTrue(std::abs(traceBaseline - traceTest) < kEpsilon,
				Widen(tfm::format("Elements are not equal: %.10f should be %.10f", traceTest, traceBaseline)).c_str());
		}

		TEST_METHOD(TestMat4Multiply)
		{
			const mat4 matrixA = { {0.3113761158140893f, 0.5660040004070195f, 0.6271553449468614f, 0.1293134122192845f}, 
				{0.7916359845436349f, 0.6544086319924736f, 0.10678317093971867f, 0.3618092618656048f}, 
				{0.8720116193879339f, 0.2942152063498533f, 0.7891131909779359f, 0.2150457540715247f}, 
				{0.8977821251253189f, 0.1869049154486111f, 0.667945907017713f, 0.3709102733463625f} };

			const mat4 matrixB = { {0.21709424438161395f, 0.31685194205855693f, 0.5740753897085578f, 0.3701079839305972f}, 
				{0.5207801280123863f, 0.7552755042870611f, 0.8083408549549509f, 0.8408666113487815f}, 
				{0.7504516274005462f, 0.7413136904674589f, 0.7893358574823457f, 0.36270511873962796f}, 
				{0.6601124044033353f, 0.11155605858385464f, 0.9855017648459508f, 0.05381918052501655f}};

			const mat4 matrixProdBaseline = { {0.9183727350786265f, 1.0054936216904127f, 1.2587523205674667f, 0.8256086480338769f}, 
				{0.8316330132274493f, 0.8646120503860107f, 1.424295421313452f, 0.9014642477423138f}, 
				{1.0766757845785908f, 1.1074821819853198f, 1.5732298890834215f, 0.8679231858785799f}, 
				{1.0383432631604148f, 0.9621634477073111f, 1.5592438867536915f, 0.7516679217130823f}};

			const mat4 matrixProdTest = matrixA * matrixB;

			TestMatIsEqual(matrixProdBaseline, matrixProdTest, 1e-6f);
		}

		TEST_METHOD(TestMat4Transpose)
		{
			const mat4 matrix = {{0.44447961821288695, - 0.7811027183432904, - 0.05859455523361223, 0.07116365077185227}, 
				{0.16635529541217453, - 0.4121151550232658, - 0.6696910890219172, 0.20251561592520462}, 
				{0.5084351515108025, 0.5422464064914099, 0.5571481714953079, - 0.9527792285283185}, 
				{0.8455132256224434, 0.9849078824530442, - 0.2991824613664673, - 0.9099905696731168}};

			const mat4 transMatrixBaseline = { {0.44447961821288695, 0.16635529541217453, 0.5084351515108025, 0.8455132256224434}, 
				{-0.7811027183432904, - 0.4121151550232658, 0.5422464064914099, 0.9849078824530442}, 
				{-0.05859455523361223, - 0.6696910890219172, 0.5571481714953079, - 0.2991824613664673}, 
				{0.07116365077185227, 0.20251561592520462, - 0.9527792285283185, - 0.9099905696731168}};

			const mat4 transMatrixTest = transpose(matrix);

			TestMatIsEqual(transMatrixBaseline, transMatrixTest, 1e-6f);
		}
		TEST_METHOD(TestMat4Determinant)
		{
			// Find the determinant of a 4x4 matrix

			const mat4 matrix = {{-0.042892382162663445f, -0.9826061663646066f, -0.3059414092901469f, -0.721440345231549f},
				{-0.6387946708850847f, 0.05740242248262417f, 0.1571735571969195f, 0.5206988374203476f},
				{-0.19140428651791153f, 0.8074525611754817f, 0.9518084961653956f, 0.24072003786258245f},
				{-0.8039970294341385f, 0.9635094414414023f, 0.5003453408886678f, -0.7325583535181814f}};

			const float matrixDetBaseline = 0.8444872131461514f;

			const float matrixDetTest = det(matrix);

			constexpr float kEpsilon = 1e-6f;
			Assert::IsTrue(std::abs(matrixDetBaseline - matrixDetTest) < kEpsilon,
						   Widen(tfm::format("Elements are not equal: %.10f should be %.10f", matrixDetBaseline, matrixDetTest)).c_str());
		}

		TEST_METHOD(TestMat4Inverse)
		{
			// Inverse of a non-singular 4x4 matrix

			const mat4 matrix = {{1.4521684705132136, -0.6657411666062907, 1.3685779839015542, -0.4365905598819211}, 
				{-0.2759170976270484, -0.8027837214736566, 0.626739861528872, -0.30653647979191767}, 
				{0.188018244645761, 1.1014213458267346, - 0.3645220628376178, 1.2459706444488443}, 
				{0.26855194801984617, - 0.2575156574079758, 1.9315152852223294, 1.475488818465541}};

			const mat4 invMatrixBaseline = { {1.3148479253618026f, 4.016455937035635f, 3.347357371817632f, - 1.6031762058032193f}, 
				{-1.784862121403524f, - 11.106159642649597f, - 7.456539926527326f, 3.4611802949237953f}, 
				{-1.2051522157233991f, - 7.418065213919103f, - 5.494144614285622f, 2.7417877905821793f}, 
				{1.0268020959882984f, 7.041374458251646f, 5.281577928682822f, - 2.0155766529602746f}};

			const mat4 invMatrixTest = inverse(matrix);

			TestMatIsEqual(invMatrixBaseline, invMatrixTest, 1e-5f);
		}

		TEST_METHOD(TestMat4SingularInverse)
		{
			// Inverse of a singular matrix. Since this has no inverse, the function should just return null (a zero matrix)

			const mat4 matrix = { {1.0f, 0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, -1.0f} };

			const mat4 invMatrixTest = inverse(matrix);

			TestMatIsEqual(mat4::Null(), invMatrixTest, 1e-5f);
		}
	};
	
	TEST_CLASS(CudaMat3Tests), MatrixTestUtils
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
			
			const mat3 matrix = { {0.6347789803421424f, - 0.7771607777375271f, 0.5790519892677031f}, 
								  {-0.6243937065879472f, - 0.5172780650846991f, -0.8685224809824379f}, 
								  {0.08449324101924827f, - 0.5376909865279451f, - 0.20798783690282585f}};

			const mat3 invMatrixBaseline = { {-2.40364128057734f, - 3.163249205240259f, 6.517303140808326f}, 
											 {-1.3592905338495636f, - 1.210164984782027f, 1.2690915550313424f}, 
											 {2.5375850527577772f, 1.8434810070449061f, - 5.441236574213772f}};			
			
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
