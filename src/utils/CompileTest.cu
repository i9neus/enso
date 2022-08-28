#include "kernels/math/CudaMath.cuh"

/*#include "kernels/CudaAsset.cuh"

#include "kernels/CudaVector.cuh"
#include "kernels/CudaSampler.cuh"

    #include "kernels/ml/CudaGrid2Grid.cuh"
    #include "kernels/ml/CudaFCNNProbeDenoiser.cuh"

#include "kernels/lights/CudaLight.cuh"
    #include "kernels/lights/CudaDistantLight.cuh"
    #include "kernels/lights/CudaEnvironmentLight.cuh"
    #include "kernels/lights/CudaQuadLight.cuh"
    #include "kernels/lights/CudaSphereLight.cuh"

#include "kernels/materials/CudaMaterial.cuh"
    #include "kernels/materials/CudaCornellMaterial.cuh"
    #include "kernels/materials/CudaEmitterMaterial.cuh"
    #include "kernels/materials/CudaKIFSMaterial.cuh"
    #include "kernels/materials/CudaSimpleMaterial.cuh"

#include "kernels/tracables/CudaTracable.cuh"
    #include "kernels/tracables/CudaBox.cuh"
    #include "kernels/tracables/CudaCornellBox.cuh"
    #include "kernels/tracables/CudaGenericIntersectors.cuh"
    #include "kernels/tracables/CudaKIFS.cuh"
    #include "kernels/tracables/CudaPlane.cuh"
    #include "kernels/tracables/CudaSDF.cuh"
    #include "kernels/tracables/CudaSphere.cuh"
*/
//#include "kernels/CudaRenderObject.cuh"
//#include "kernels/CudaRenderObjectContainer.cuh"
#include "kernels/CudaWavefrontTracer.cuh"
//#include "kernels/GlobalResourceRegistry.cuh"

__device__ void CompileTestDevice()
{
    int a = max(1, 2);
}

__host__ void CompileTestHost()
{
    int a = max(1, 2);
}