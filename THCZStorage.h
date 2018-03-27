#ifndef THCZ_STORAGE_INC
#define THCZ_STORAGE_INC

#include "THZ/THZStorage.h"
#include "THC/THCStorage.h"
#include "THCZGeneral.h"

#define THCZStorage        TH_CONCAT_3(TH,CNType,Storage)
#define THCZStorage_(NAME) TH_CONCAT_4(TH,CNType,Storage_,NAME)

#include "generic/THCZStorage.h"
#include "THCZGenerateAllTypes.h"

#endif
