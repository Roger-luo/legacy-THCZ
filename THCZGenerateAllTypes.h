#ifndef THCZ_GENERIC_FILE
#error "You must define THCZ_GENERIC_FILE before including THCZGenerateAllTypes.h"
#endif

#define THCZGenerateAllTypes

#define THCZTypeIdxByte   1
#define THCZTypeIdxChar   2
#define THCZTypeIdxShort  3
#define THCZTypeIdxInt    4
#define THCZTypeIdxLong   5
#define THCZTypeIdxFloat  6
#define THCZTypeIdxDouble 7
#define THCZTypeIdxHalf   8
#define THCZTypeIdxZFloat  9
#define THCZTypeIdxZDouble 10
#define THCZTypeIdx_(T) TH_CONCAT_2(THCZTypeIdx,T)

#include "THCZGenerateZFloatType.h"
#include "THCZGenerateZDoubleType.h"

#undef THCZTypeIdxZFloat
#undef THCZTypeIdxZDouble
#undef THCZTypeIdx_

#undef THCZGenerateAllTypes
#undef THCZ_GENERIC_FILE
