#ifndef THCZ_TENSOR_INC
#define THCZ_TENSOR_INC

#include "TH/THTensor.h"
#include "THZ/THZTensor.h"
#include "THC/THCTensor.h"
#include "THCZStorage.h"
#include "THCZGeneral.h"

#define THCZTensor          TH_CONCAT_3(TH,CNType,Tensor)
#define THCZTensor_(NAME)   TH_CONCAT_4(TH,CNType,Tensor_,NAME)

#include "generic/THCZTensor.h"
#include "THCZGenerateAllTypes.h"

#endif
