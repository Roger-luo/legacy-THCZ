#include "THCZStorage.h"

#include "THC/THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

#include "generic/THCZStorage.cu"
#include "THCZGenerateAllTypes.h"