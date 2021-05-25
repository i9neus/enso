#include "CppUnitTest.h"
#include "generic\StdIncludes.h"
#include "kernels\CudaMath.cuh"
#include "generic\StringUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Cuda;

namespace tests
{
	class MatrixTestUtils
	{
	public: 
		template<typename T>
		void TestIsEqual(const T& matBaseline, const T& matTest, const float kEpsilon)
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

			TestIsEqual(matrixProdBaseline, matrixProdTest, 1e-6f);
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

			TestIsEqual(transMatrixBaseline, transMatrixTest, 1e-6f);
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

			TestIsEqual(invMatrixBaseline, invMatrixTest, 1e-5f);
		}

		TEST_METHOD(TestMat4SingularInverse)
		{
			// Inverse of a singular matrix. Since this has no inverse, the function should just return null (a zero matrix)

			const mat4 matrix = { {1.0f, 0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f, -1.0f} };

			const mat4 invMatrixTest = inverse(matrix);

			TestIsEqual(mat4::null(), invMatrixTest, 1e-5f);
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

			TestIsEqual(matrixProdBaseline, matrixProdTest, 1e-6f);
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

			TestIsEqual(transMatrixBaseline, transMatrixTest, 1e-6f);
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

			TestIsEqual(invMatrixBaseline, invMatrixTest, 1e-5f);
		}

		TEST_METHOD(TestMat3SingularInverse)
		{
			// Inverse of a singular matrix. Since this has no inverse, the function should just return null (a zero matrix)

			const mat3 matrix = { {1.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 1.0f} };

			const mat3 invMatrixTest = inverse(matrix);

			TestIsEqual(mat3::null(), invMatrixTest, 1e-5f);
		}
	};
}
