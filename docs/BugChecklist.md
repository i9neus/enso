# CUDA crash/bug checklist

## Error 700, illegal address

- Do all objects with explicit ctors and that are instantiated on the device have a __device__ specifier in the ctor? 
  Nvcc fails to detect nested objects where some objects don't have __device__

## Host segfault:

- Are you upcasting to a virtually inherited class? Upcasting a device pointer on the host triggers a vtable lookup and will cause a segfault.
