#include "CudaGeodesics.cuh"

namespace Cuda
{
    class Tri
    {
    public:
        Tri(const vec3& v0, const vec3& v1, const vec3& v2) : m_v({ v0, v1, v2 }) {}
        vec3& operator[](const int idx) { return m_v[idx]; }

    private:
        std::array<vec3, 3> m_v;
    };
    
    __host__ void ConstructTetrahedtron(std::vector<Tri>& output)
    {
        static const vec3 V[4] = { normalize(vec3(1, 1, 1)), normalize(vec3(-1, -1, 1)), normalize(vec3(1, -1, -1)), normalize(vec3(-1, 1, -1)) };
        static const int I[12] = { 0, 2, 1, 1, 3, 2, 2, 0, 3, 3, 1, 0 };    

        for (int i = 0; i < 4; ++i) 
        { 
            output.emplace_back(V[I[i * 3]], V[I[i * 3 + 1]], V[I[i * 3 + 2]]); 
        }
    }

    __host__ void ConstructIcosahedron(std::vector<Tri>& output)
    {
        // Set up some vertex buffers
        vec3 ringA[5], ringB[5];
        for (int i = 0; i < 5; ++i)
        {
            float theta = kHalfPi - std::atan(0.5f);
            float phi = kTwoPi * i / 5.0f;
            ringA[i] = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));

            theta = kPi - theta;
            phi = kTwoPi * (i + 0.5f) / 5.0f;
            ringB[i] = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        }

        for (int i = 0; i < 5; ++i)
        {
            // Top and bottom caps
            output.emplace_back(vec3(0.0f, 0.0f, 1.0f), ringA[i], ringA[(i + 1) % 5]);
            output.emplace_back(vec3(0.0f, 0.0f, -1.0f), ringB[i], ringB[(i + 1) % 5]);
            
            // Triangle strip around the equator
            output.emplace_back(ringA[i], ringA[(i + 1) % 5], ringB[i]);
            output.emplace_back(ringB[i], ringB[(i + 1) % 5], ringA[(i + 1) % 5]);
        }
    }
    
    __host__ void GenerateGeodesicDistribution(std::vector<vec3>& output, const GeodesicClass geoClass, int numSubDivs)
    {
        // Cap the number of iterations to prevent an explosion in the number of samples
        constexpr int kMaxSubDivs = 10;
        if (numSubDivs > kMaxSubDivs)
        {
            Log::ErrorOnce("Exceeded the maximum number of iterations for a geodesic distribution. Clamping to %i.", kMaxSubDivs);
            numSubDivs = ::min(numSubDivs, kMaxSubDivs);
        }

        std::array<std::vector<Tri>, 2> listBuffers;
        std::vector<Tri>* sourceList = &listBuffers[0], * targetList = &listBuffers[1];

        // Create the Platonic primitive
        switch (geoClass)
        {
        case kGeoClassTetrahedron:
            ConstructTetrahedtron(*targetList); break;
        case kGeoClassIcosahedron:
            ConstructIcosahedron(*targetList); break;
        default:
            AssertMsg(false, "Invalid geodesic class.");
        }

        output.clear();
        for (int subDivIdx = 0; subDivIdx < numSubDivs; ++subDivIdx)
        {
            std::swap(sourceList, targetList);
            targetList->clear();
            for (auto& t : *sourceList)
            {
                const vec3 v01 = mix(t[0], t[1], 0.5f);
                const vec3 v12 = mix(t[1], t[2], 0.5f);
                const vec3 v20 = mix(t[2], t[0], 0.5f);
                const vec3 vc = (t[0] + t[1] + t[2]) / 3.0f;

                targetList->emplace_back(t[0], v01, v20);
                targetList->emplace_back(t[1], v01, v12);
                targetList->emplace_back(t[2], v12, v20);
                targetList->emplace_back(v01, v12, v20);
            }

            // Project each vertex onto the sphere
            for (auto& t : *targetList)
            {
                for (int j = 0; j < 3; ++j)
                {
                    t[j] = normalize(t[j]);
                }
            }
        }

        // Copy to the output buffer
        for (auto& t : *targetList)
        {
            const vec3 c = (t[0] + t[1] + t[2]) / 3.0f;
            output.push_back(normalize(c));
        }
    }
}