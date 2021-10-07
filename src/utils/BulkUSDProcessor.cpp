#include "BulkUSDProcessor.h"

#include "generic/StringUtils.h"
#include "generic/JsonUtils.h"

#include "io/USDIO.h"
#include "kernels/lightprobes/CudaLightProbeGrid.cuh"
#include "kernels/math/CudaSphericalHarmonics.cuh"

#include <filesystem>
#include <random>

void BulkConvertUSDProbeGrds()
{
	namespace fs = std::filesystem;
	const std::string inputRoot = fs::path("C:/projects/probegen/data/").string();
	const std::string outputRoot = fs::path("C:/projects/probegen/data/remapped/").string();

	auto createOutputPath = [&](const std::string& inputPath) -> std::string
	{
		return fs::path(outputRoot + inputPath.substr(inputRoot.size(), inputPath.size() - inputRoot.size())).string();
	};

	// Create some output directories based on the contents of the input root
	/*if (!fs::is_directory(outputRoot))
	{
		for (auto const& entry : fs::recursive_directory_iterator(inputRoot))
		{
			if (!entry.is_directory()) { continue; }

			const std::string inputPath = entry.path().string();
			const std::string outputPath = createOutputPath(inputPath);

			if (fs::is_directory(outputPath))
			{
				Log::Debug("'%s' already exists.", outputPath);
			}
			else
			{
				try
				{
					//fs::create_directories(fs::path(outputPath));
					Log::Warning("Created directory at '%s'", outputPath);
				}
				catch (std::runtime_error& err)
				{
					Log::Error("Error creating directory: %s", err.what());
				}
			}
		}
	}*/

	std::vector<Cuda::vec3> probeData;
	Cuda::LightProbeGridParams gridParams;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<> rng(0, std::numeric_limits<int>::max());

	struct ID
	{
		ID() = default;
		std::string id;
		bool isValid = true;
	};

	std::map<std::string, std::map<std::string, ID>> idMap;

	// Precompute some coefficients
	const float sqrtPI = std::sqrt(kPi);
	const float fC0 = 1.0f / (2.0f * sqrtPI);
	const float fC1 = std::sqrt(3.0f) / (3.0f * sqrtPI);	

	int totalGrids = 0, validGrids = 0, errors = 0;
	for (auto const& entry : fs::recursive_directory_iterator(inputRoot))
	{
		if (!entry.is_regular_file()) { continue; }

		// Don't search in the output directory
		if (entry.path().parent_path() == fs::path(outputRoot).string()) { continue; }

		// Verify that we can actually load this file
		auto extension = entry.path().extension().string();
		if (extension != ".usd" && extension != ".usda") { continue; }

		// Get the input path and parent directory
		const std::string inputPath = entry.path().string();
		const std::string parentDirectory = entry.path().parent_path().string();
		
		// Get the input filename and strip out the spaces
		std::string inputFilename = entry.path().filename().string();
		std::string temp;
		for (const char c : inputFilename) { if (c != ' ') { temp += c; } }
		inputFilename = temp;

		//SceneA32.direct.1781884779.034504
		// Scene201742xxx.direct.181316516612100000.usd
		// Scene864736xxx.direct.0443066935  100000.usd

		// Decompose the filename into its constitent parts
		std::string prefix, sampleCountStr, component, randomDigits, seed;
		Lexer lex(inputFilename);
		if (!lex.ParseToken(prefix, [](const char c) { return std::isalpha(c); })) { continue; }
		if (!lex.ParseToken(sampleCountStr, [](const char c) { return std::isdigit(c); })) { continue; }
		if (!lex.ReadNext('.')) { continue; }
		if (!lex.ParseToken(component, [](const char c) { return std::isalpha(c); })) { continue; }				

		const int sampleCount = std::atoi(sampleCountStr.c_str());
		Assert(sampleCount > 0 && sampleCount < 10000000);

		totalGrids++;

		Log::Write("Processing %s...", inputPath);
		{
			Log::Indent indent;

			// Check to see if we've seen a prefix with this name before
			auto& newID = idMap[parentDirectory][prefix];
			auto& newPrefix = newID.id;
			if (newPrefix.empty())
			{
				newPrefix = tfm::format("Scene%06i", rng(mt) % 1000000);
				Log::Warning("New scene: %s", newPrefix);
			}

			// Compose the new filename		
			auto newFilename = tfm::format("%s.%s.%06i%06i%06i%s", newPrefix, component, rng(mt) % 1000000, rng(mt) % 1000000, sampleCount, extension);

			// Generate an output path
			//std::string outputPath = fs::path(createOutputPath(inputPath)).replace_filename(fs::path(newFilename)).string();
			std::string outputPath = outputRoot + newFilename;

			// Read in the probe volume
			try
			{
				USDIO::ReadGridDataUSD(probeData, gridParams, inputPath, USDIO::SHPackingFormat::kNone);
			}
			catch (std::runtime_error& err)
			{
				Log::Error("Error reading USD file: %s", err.what());
			}

			// Sanity check
			Assert(probeData.size() == gridParams.numProbes * gridParams.coefficientsPerProbe);
			Assert(gridParams.coefficientsPerProbe == 5);			

			// Pre-multiply the coefficients
			double meanValidity = 0.0;
			Cuda::vec3 C[5], D[5];
			for (int probeIdx = 0, dataIdx = 0; probeIdx < gridParams.numProbes; ++probeIdx, dataIdx += gridParams.coefficientsPerProbe)
			{				
				for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe; ++coeffIdx) { C[coeffIdx] = probeData[dataIdx + coeffIdx]; }

				// Pre-multiply the coefficients
				C[0] *= fC0;
				C[1] *= -fC1;
				C[2] *= fC1;
				C[3] *= -fC1;

				// Pack the coefficients using Unity's perferred format
				D[0] = C[0];
				D[1] = Cuda::vec3(C[1].x, C[2].x, C[3].x);
				D[2] = Cuda::vec3(C[1].y, C[2].y, C[3].y);
				D[3] = Cuda::vec3(C[1].z, C[2].z, C[3].z);
				D[4] = C[4];

				// Accumulate validity
				meanValidity += C[4].x;

				//memcpy(D, C, sizeof(Cuda::vec3) * 5);

				for (int coeffIdx = 0; coeffIdx < gridParams.coefficientsPerProbe; ++coeffIdx) { probeData[dataIdx + coeffIdx] = D[coeffIdx]; }				
			}

			meanValidity /= double(gridParams.numProbes);
			if (meanValidity < 0.7)
			{
				Log::Warning("Probe grid '%s' has only %f%% validity. Skipping...\n", newFilename, meanValidity * 100.0);
				newID.isValid = false;
				continue;
			}

			// Write out the probe volume
			try
			{
				USDIO::WriteGridDataUSD(probeData, gridParams, outputPath, USDIO::SHPackingFormat::kUnity);
			}
			catch (std::runtime_error& err)
			{
				Log::Error("Error reading USD file: %s", err.what());
			}
			validGrids++;

			if (!newID.isValid) { errors++; }
		}
	}

	Log::NL;
	Log::Write("Finished processing %i grids: %i exported, %i skipped, %i errors", totalGrids, validGrids, totalGrids - validGrids, errors);

	//std::vector<Cuda::vec3> gridData;
	//Cuda::LightProbeGridParams gridParams;
	//USDIO::ReadGridDataUSD(gridData, gridParams, "C:/projects/probegen/data/set7/grids/SceneH100000.direct. 743799997.762703.usd");
}