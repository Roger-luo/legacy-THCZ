set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

set(extra_src)
# loop over all types
# foreach(THC_TYPE Byte Char Short Int Long Half Float Double)
#    # loop over files which need to be split between types (because of long compile times)
#    foreach(THC_FILE TensorSort TensorMathCompareT TensorMathPointwise TensorMathCompare TensorMathReduce TensorMasked)
#       if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
#          FILE(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu"
#               "#include \"../THC${THC_FILE}.cuh\"\n#include \"../generic/THC${THC_FILE}.cu\"\n#include \"../THCGenerate${THC_TYPE}Type.h\"\n")
#       endif()
#       LIST(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
#    endforeach()
# endforeach()

set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS}
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCCachingAllocator.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCCachingHostAllocator.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCGeneral.c
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.c
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCStream.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCZTensor.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCZTensorCopy.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorRandom.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCThreadLocal.c

  # ${CMAKE_CURRENT_SOURCE_DIR}/THCReduceApplyUtils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCZBlas.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCSleep.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCZStorage.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCZTensor.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCZTensorCopy.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMath.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathBlas.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathMagma.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathPairwise.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathReduce.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathScan.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorIndex.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorConv.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorRandom.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorScatterGather.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorTopK.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorSort.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCZTensorTypeUtils.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCSortUtils.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMode.cu
  # ${extra_src}
  PARENT_SCOPE)

INSTALL(FILES
          THCZ.h
          THCZGeneral.h
          THCZBlas.h
          # THCSleep.h
          THCZStorage.h
          # THCStorageCopy.h
          # THCStream.h
          # THCThreadLocal.h
          # THCZTensor.h
          # THCZTensorCopy.h
          # THCTensorRandom.h
          # THCTensorMath.h
          # THCTensorConv.h
          # THCApply.cuh
          # THCReduce.cuh
          # THCReduceAll.cuh
          # THCReduceApplyUtils.cuh
          # THCTensorMathReduce.cuh
          # THCAsmUtils.cuh
          # THCAtomics.cuh
          # THCScanUtils.cuh
          # THCSortUtils.cuh
          # THCAllocator.h
          # THCCachingAllocator.h
          # THCCachingHostAllocator.h
          # THCDeviceUtils.cuh
          # THCDeviceTensor.cuh
          # THCDeviceTensor-inl.cuh
          # THCDeviceTensorUtils.cuh
          # THCDeviceTensorUtils-inl.cuh
          THCZGenerateAllTypes.h
          # THCGenerateByteType.h
          # THCGenerateCharType.h
          # THCGenerateShortType.h
          # THCGenerateIntType.h
          # THCGenerateLongType.h
          # THCGenerateHalfType.h
          # THCGenerateFloatType.h
          THCZGenerateZFloatTypes.h
          THCZGenerateZDoubleType.h
          # THCHalf.h
          # THCZNumerics.cuh
          # THCTensorSort.cuh
          # THCTensorInfo.cuh
          # THCTensorMathPointwise.cuh
          # THCZTensorTypeUtils.cuh
          # THCTensorRandom.cuh
          # THCTensorMathMagma.cuh
          # THCThrustAllocator.cuh
          # THCTensorMode.cuh
          # THCTensorTopK.cuh
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THCZ")

INSTALL(FILES
          generic/THCZStorage.cpp
          generic/THCZStorage.cu
          generic/THCZStorage.h
          # generic/THCZTensor.cpp
          # generic/THCZTensor.cu
          # generic/THCZTensor.h
          # generic/THCStorageCopy.c
          # generic/THCStorageCopy.cu
          # generic/THCStorageCopy.h
          # generic/THCZTensorCopy.cpp
          # generic/THCZTensorCopy.cu
          # generic/THCZTensorCopy.h
          # generic/THCTensorMasked.h
          # generic/THCTensorMasked.cu
          # generic/THCTensorMath.h
          # generic/THCTensorMath.cu
          # generic/THCTensorMathBlas.cu
          # generic/THCTensorMathBlas.h
          # generic/THCTensorMathCompare.h
          # generic/THCTensorMathCompare.cu
          # generic/THCTensorMathCompareT.h
          # generic/THCTensorMathCompareT.cu
          # generic/THCTensorMathMagma.h
          # generic/THCTensorMathMagma.cu
          # generic/THCTensorMathPairwise.h
          # generic/THCTensorMathPairwise.cu
          # generic/THCTensorMathPointwise.h
          # generic/THCTensorMathPointwise.cu
          # generic/THCTensorMathReduce.h
          # generic/THCTensorMathReduce.cu
          # generic/THCTensorMathScan.h
          # generic/THCTensorMathScan.cu
          # generic/THCTensorScatterGather.h
          # generic/THCTensorScatterGather.cu
          # generic/THCTensorIndex.h
          # generic/THCTensorIndex.cu
          # generic/THCTensorSort.h
          # generic/THCTensorSort.cu
          # generic/THCDeviceTensorUtils.cu
          # generic/THCTensorRandom.h
          # generic/THCTensorRandom.cu
          # generic/THCTensorMode.h
          # generic/THCTensorMode.cu
          # generic/THCTensorTopK.h
          # generic/THCTensorTopK.cu
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THCZ/generic")
