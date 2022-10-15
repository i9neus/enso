#pragma once

#include "SuiteBase.h"

namespace Tests
{
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
			const vec3 b0 = {1.775825603536588, 0.7341684559831378, -0.12139078960623273};
			const float dotBaseline = 0.7816439441941685f;
			const float dotTest = dot(a0, b0);

			constexpr float kEpsilon = 1e-7f;
			Assert::IsTrue(abs(dotBaseline - dotTest) < kEpsilon,
				Widen(tfm::format("Elements are not equal: %.10f should be %.10f", dotBaseline, dotTest)).c_str());

			const vec3 a1 = {1.775825603536588f, 0.7341684559831378f, -0.12139078960623273f};
			const vec3 b1 = {-0.37584680565822204f, 0.7414234438824758f, -0.6135374960846884f};
			const vec3 crossBaseline = {-0.36043789890279004f, 1.1351599347867507f, 1.5925736037050646f};
			const vec3 crossTest = cross(a1, b1);

			Assert::IsTrue(cwiseMax(abs(crossBaseline - crossTest)) < kEpsilon,
				Widen(tfm::format("Elements are not equal: %s should be %s", crossTest.format(), crossBaseline.format())).c_str());
		}

		TEST_METHOD(TestVec4Arithmetic)
		{
			const vec4 a = {1.4521684705132136f, -0.6657411666062907f, 1.3685779839015542f, -0.4365905598819211f };
			const vec4 b = {-0.2759170976270484f, -0.8027837214736566f, 0.626739861528872f, -0.30653647979191767f };
			const vec4 c = {0.188018244645761f, 1.1014213458267346f, -0.3645220628376178f, 1.2459706444488443f };
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
}