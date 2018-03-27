#ifndef THCZ_TENSOR_TYPE_UTILS_INC
#define THCZ_TENSOR_TYPE_UTILS_INC

#include "THCZTensor.h"
#include "THC/THCTensorTypeUtils.cuh"

#define TENSOR_UTILS(TENSOR_TYPE, DATA_TYPE, ACC_DATA_TYPE)             \
  template <>                                                           \
  struct THC_CLASS TensorUtils<TENSOR_TYPE> {                                     \
    typedef DATA_TYPE DataType;                                         \
    typedef ACC_DATA_TYPE AccDataType;                                  \
                                                                        \
    static TENSOR_TYPE* newTensor(THCState* state);                     \
    static TENSOR_TYPE* newContiguous(THCState* state, TENSOR_TYPE* t); \
    static THLongStorage* newSizeOf(THCState* state, TENSOR_TYPE* t);   \
    static void retain(THCState* state, TENSOR_TYPE* t);                \
    static void free(THCState* state, TENSOR_TYPE* t);                  \
    static void freeCopyTo(THCState* state, TENSOR_TYPE* src,           \
                           TENSOR_TYPE* dst);                           \
    static void resize(THCState* state, TENSOR_TYPE* out,               \
                       THLongStorage* sizes,                            \
                       THLongStorage* strides);                         \
    static void resizeAs(THCState* state, TENSOR_TYPE* dst,             \
                         TENSOR_TYPE* src);                             \
    static void squeeze1d(THCState *state, TENSOR_TYPE *dst,            \
                          TENSOR_TYPE *src, int dimension);             \
    static DATA_TYPE* getData(THCState* state, TENSOR_TYPE* t);         \
    static ptrdiff_t getNumElements(THCState* state, TENSOR_TYPE* t);   \
    static int64_t getSize(THCState* state, TENSOR_TYPE* t, int dim);   \
    static int64_t getStride(THCState* state, TENSOR_TYPE* t, int dim); \
    static int getDims(THCState* state, TENSOR_TYPE* t);                \
    static bool isContiguous(THCState* state, TENSOR_TYPE* t);          \
    static bool allContiguous(THCState* state, TENSOR_TYPE** inputs, int numInputs); \
    static int getDevice(THCState* state, TENSOR_TYPE* t);              \
    static bool allSameDevice(THCState* state, TENSOR_TYPE** inputs, int numInputs); \
    static void copyIgnoringOverlaps(THCState* state,                   \
                                     TENSOR_TYPE* dst, TENSOR_TYPE* src); \
    /* Determines if the given tensor has overlapping data points (i.e., */ \
    /* is there more than one index into the tensor that references */  \
    /* the same piece of data)? */                                      \
    static bool overlappingIndices(THCState* state, TENSOR_TYPE* t);    \
    /* Can we use 32 bit math for indexing? */                          \
    static bool canUse32BitIndexMath(THCState* state, TENSOR_TYPE* t, ptrdiff_t max_elem=UINT32_MAX);  \
    /* Are all tensors 32-bit indexable? */                             \
    static bool all32BitIndexable(THCState* state, TENSOR_TYPE** inputs, int numInputs); \
  }

TENSOR_UTILS(THCudaZFloatTensor, thrust::complex<float>, thrust::complex<float>);
TENSOR_UTILS(THCudaZDoubleTensor, thrust::complex<double>, thrust::complex<double>);

#undef TENSOR_UTILS

#endif // THC_TENSOR_TYPE_UTILS_INC