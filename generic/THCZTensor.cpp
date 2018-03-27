#ifndef THCZ_GENERIC_FILE
#define THCZ_GENERIC_FILE "generic/THCZTensor.cpp"
#else

/**** access methods ****/
THCZStorage *THCZTensor_(storage)(THCState *state, const THCZTensor *self)
{
  return self->storage;
}

ptrdiff_t THCZTensor_(storageOffset)(THCState *state, const THCZTensor *self)
{
  return self->storageOffset;
}

int THCZTensor_(nDimension)(THCState *state, const THCZTensor *self)
{
  return self->nDimension;
}

int64_t THCZTensor_(size)(THCState *state, const THCZTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

int64_t THCZTensor_(stride)(THCState *state, const THCZTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}

THLongStorage *THCZTensor_(newSizeOf)(THCState *state, THCZTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THCZTensor_(newStrideOf)(THCState *state, THCZTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

ntype *THCZTensor_(data)(THCState *state, const THCZTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

void THCZTensor_(setFlag)(THCState *state, THCZTensor *self, const char flag)
{
  self->flag |= flag;
}

void THCZTensor_(clearFlag)(THCState *state, THCZTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THCZTensor_(rawInit)(THCState *state, THCZTensor *self);


/* Empty init */
THCZTensor *THCZTensor_(new)(THCState *state)
{
  THCZTensor *self = (THCZTensor*)THAlloc(sizeof(THCZTensor));
  THCZTensor_(rawInit)(state, self);
  return self;
}

/* Pointer-copy init */
THCZTensor *THCZTensor_(newWithTensor)(THCState *state, THCZTensor *tensor)
{
  THCZTensor *self = (THCZTensor*)THAlloc(sizeof(THCZTensor));
  THCZTensor_(rawInit)(state, self);
  THCZTensor_(setStorageNd)(state,
                           self,
                           tensor->storage,
                           tensor->storageOffset,
                           tensor->nDimension,
                           tensor->size,
                           tensor->stride);
  return self;
}

/* Storage init */
THCZTensor *THCZTensor_(newWithStorage)(THCState *state, THCZStorage *storage, ptrdiff_t storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THCZTensor *self = (THCZTensor*)THAlloc(sizeof(THCZTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THCZTensor_(rawInit)(state, self);
  THCZTensor_(setStorageNd)(state,
                           self,
                           storage,
                           storageOffset,
                           (size ? size->size : (stride ? stride->size : 0)),
                           (size ? size->data : NULL),
                           (stride ? stride->data : NULL));

  return self;
}
THCZTensor *THCZTensor_(newWithStorage1d)(THCState *state, THCZStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0)
{
  return THCZTensor_(newWithStorage4d)(state, storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THCZTensor *THCZTensor_(newWithStorage2d)(THCState *state, THCZStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1)
{
  return THCZTensor_(newWithStorage4d)(state, storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THCZTensor *THCZTensor_(newWithStorage3d)(THCState *state, THCZStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2)
{
  return THCZTensor_(newWithStorage4d)(state, storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THCZTensor *THCZTensor_(newWithStorage4d)(THCState *state, THCZStorage *storage, ptrdiff_t storageOffset,
                               int64_t size0, int64_t stride0,
                               int64_t size1, int64_t stride1,
                               int64_t size2, int64_t stride2,
                               int64_t size3, int64_t stride3)
{
  int64_t size[4] = {size0, size1, size2, size3};
  int64_t stride[4] = {stride0, stride1, stride2, stride3};

  THCZTensor *self = (THCZTensor*)THAlloc(sizeof(THCZTensor));
  THCZTensor_(rawInit)(state, self);
  THCZTensor_(setStorageNd)(state, self, storage, storageOffset, 4, size, stride);

  return self;
}

THCZTensor *THCZTensor_(newWithSize)(THCState *state, THLongStorage *size, THLongStorage *stride)
{
  return THCZTensor_(newWithStorage)(state, NULL, 0, size, stride);
}

THCZTensor *THCZTensor_(newWithSize1d)(THCState *state, int64_t size0)
{
  return THCZTensor_(newWithSize4d)(state, size0, -1, -1, -1);
}

THCZTensor *THCZTensor_(newWithSize2d)(THCState *state, int64_t size0, int64_t size1)
{
  return THCZTensor_(newWithSize4d)(state, size0, size1, -1, -1);
}

THCZTensor *THCZTensor_(newWithSize3d)(THCState *state, int64_t size0, int64_t size1, int64_t size2)
{
  return THCZTensor_(newWithSize4d)(state, size0, size1, size2, -1);
}

THCZTensor *THCZTensor_(newWithSize4d)(THCState *state, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};

  THCZTensor *self = (THCZTensor*)THAlloc(sizeof(THCZTensor));
  THCZTensor_(rawInit)(state, self);
  THCZTensor_(resizeNd)(state, self, 4, size, NULL);

  return self;
}

THCZTensor *THCZTensor_(newClone)(THCState *state, THCZTensor *self)
{
  THCZTensor *tensor = THCZTensor_(new)(state);
  THCZTensor_(resizeAs)(state, tensor, self);
  THCZTensor_(copy)(state, tensor, self);
  return tensor;
}

THCZTensor *THCZTensor_(newContiguous)(THCState *state, THCZTensor *self)
{
  if(!THCZTensor_(isContiguous)(state, self)) {
    return THCZTensor_(newClone)(state, self);
  } else {
    THCZTensor_(retain)(state, self);
    return self;
  }
}

THCZTensor *THCZTensor_(newSelect)(THCState *state, THCZTensor *tensor, int dimension_, int64_t sliceIndex_)
{
  THCZTensor *self = THCZTensor_(newWithTensor)(state, tensor);
  THCZTensor_(select)(state, self, NULL, dimension_, sliceIndex_);
  return self;
}

THCZTensor *THCZTensor_(newNarrow)(THCState *state, THCZTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_)
{
  THCZTensor *self = THCZTensor_(newWithTensor)(state, tensor);
  THCZTensor_(narrow)(state, self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THCZTensor *THCZTensor_(newTranspose)(THCState *state, THCZTensor *tensor, int dimension1_, int dimension2_)
{
  THCZTensor *self = THCZTensor_(newWithTensor)(state, tensor);
  THCZTensor_(transpose)(state, self, NULL, dimension1_, dimension2_);
  return self;
}

THCZTensor *THCZTensor_(newUnfold)(THCState *state, THCZTensor *tensor, int dimension_, int64_t size_, int64_t step_)
{
  THCZTensor *self = THCZTensor_(newWithTensor)(state, tensor);
  THCZTensor_(unfold)(state, self, NULL, dimension_, size_, step_);
  return self;
}

THCZTensor *THCZTensor_(newView)(THCState *state, THCZTensor *tensor, THLongStorage *size)
{
  THArgCheck(THCZTensor_(isContiguous)(state, tensor), 2, "input is not contiguous");
  ptrdiff_t numel = THCZTensor_(nElement)(state, tensor);
  THCZTensor *self = THCZTensor_(new)(state);
  THLongStorage *inferred_size = THLongStorage_newInferSize(size, numel);
  THCZTensor_(setStorage)(state, self, tensor->storage, tensor->storageOffset, inferred_size, NULL);
  THLongStorage_free(inferred_size);
  return self;
}

/* Resize */
void THCZTensor_(resize)(THCState *state, THCZTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THCZTensor_(resizeNd)(state, self, size->size, size->data, (stride ? stride->data : NULL));
}

void THCZTensor_(resizeAs)(THCState *state, THCZTensor *self, THCZTensor *src)
{
  int isSame = 0;
  int d;
  if(self->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < self->nDimension; d++)
    {
      if(self->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THCZTensor_(resizeNd)(state, self, src->nDimension, src->size, NULL);
}

void THCZTensor_(resize1d)(THCState *state, THCZTensor *tensor, int64_t size0)
{
  THCZTensor_(resize4d)(state, tensor, size0, -1, -1, -1);
}

void THCZTensor_(resize2d)(THCState *state, THCZTensor *tensor, int64_t size0, int64_t size1)
{
  THCZTensor_(resize4d)(state, tensor, size0, size1, -1, -1);
}

void THCZTensor_(resize3d)(THCState *state, THCZTensor *tensor, int64_t size0, int64_t size1, int64_t size2)
{
  THCZTensor_(resize4d)(state, tensor, size0, size1, size2, -1);
}

void THCZTensor_(resize4d)(THCState *state, THCZTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3)
{
  int64_t size[4] = {size0, size1, size2, size3};

  THCZTensor_(resizeNd)(state, self, 4, size, NULL);
}

void THCZTensor_(resize5d)(THCState *state, THCZTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3, int64_t size4)
{
    int64_t size[5] = {size0, size1, size2, size3, size4};

  THCZTensor_(resizeNd)(state, self, 5, size, NULL);
}

THCZTensor* THCZTensor_(newExpand)(THCState *state, THCZTensor *tensor, THLongStorage *sizes)
{
  THCZTensor *result = THCZTensor_(new)(state);
  THCZTensor_(expand)(state, result, tensor, sizes);
  return result;
}

void THCZTensor_(expand)(THCState *state, THCZTensor *r, THCZTensor *tensor, THLongStorage *sizes)
{
  THArgCheck(THCZTensor_(nDimension)(state, tensor) > 0 || THLongStorage_size(sizes) == 0, 0,
       "can't expand an empty tensor");
  THArgCheck(THLongStorage_size(sizes) >= THCZTensor_(nDimension)(state, tensor), 1,
             "the number of sizes provided must be greater or equal to the "
             "number of dimensions in the tensor");

  int64_t *expandedSizes;
  int64_t *expandedStrides;
  char error_buffer[1024];
  int ret = THLongStorage_inferExpandGeometry(tensor->size,
                                              tensor->stride,
                                              THCZTensor_(nDimension)(state, tensor),
                                              sizes,
                                              &expandedSizes,
                                              &expandedStrides,
                                              error_buffer,
                                              1024);
  if (ret != 0) {
    THError(error_buffer);
    return;
  }
  THCZTensor_(setStorageNd)(state, r, THCZTensor_(storage)(state, tensor), THCZTensor_(storageOffset)(state, tensor),
                           THLongStorage_size(sizes), expandedSizes, expandedStrides);
  THFree(expandedSizes);
  THFree(expandedStrides);
}

void THCZTensor_(expandNd)(THCState *state, THCZTensor **rets, THCZTensor **ops, int count) {
  for (int i = 0; i < count; ++i) {
    THArgCheck(THCZTensor_(nDimension)(state, ops[i]) > 0, i, "can't expand empty tensor %d", i);
  }

  int64_t **op_sizes = (int64_t **)THAlloc(count * sizeof(int64_t *));
  int64_t *op_dims = (int64_t *)THAlloc(count * sizeof(int64_t));

  for (int i = 0; i < count; ++i) {
    op_sizes[i] = ops[i]->size;
    op_dims[i] = ops[i]->nDimension;
  }

  THLongStorage *sizes = THLongStorage_new();
  char error_buffer[1024];
  int ret = THLongStorage_inferSizeN(sizes,
                                     count,
                                     op_sizes,
                                     op_dims,
                                     error_buffer,
                                     1024);

  if(ret != 0) {
    THLongStorage_free(sizes);
    THFree(op_sizes);
    THFree(op_dims);
    THError(error_buffer);
    return;
  }

  for (int i = 0; i < count; ++i) {
    THCZTensor_(expand)(state, rets[i], ops[i], sizes);
  }

  THLongStorage_free(sizes);
  THFree(op_sizes);
  THFree(op_dims);
}

void THCZTensor_(set)(THCState *state, THCZTensor *self, THCZTensor *src)
{
  if(self != src)
    THCZTensor_(setStorageNd)(state,
                             self,
                             src->storage,
                             src->storageOffset,
                             src->nDimension,
                             src->size,
                             src->stride);
}

void THCZTensor_(setStorage)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  THCZTensor_(setStorageNd)(state,
                           self,
                           storage_,
                           storageOffset_,
                           (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                           (size_ ? size_->data : NULL),
                           (stride_ ? stride_->data : NULL));
}

void THCZTensor_(setStorage1d)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_)
{
  THCZTensor_(setStorage4d)(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            -1, -1,
                            -1, -1,
                            -1, -1);
}

void THCZTensor_(setStorage2d)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_)
{
  THCZTensor_(setStorage4d)(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            -1, -1,
                            -1, -1);
}

void THCZTensor_(setStorage3d)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_)
{
  THCZTensor_(setStorage4d)(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            size2_, stride2_,
                            -1, -1);
}

void THCZTensor_(setStorage4d)(THCState *state, THCZTensor *self, THCZStorage *storage_, ptrdiff_t storageOffset_,
                             int64_t size0_, int64_t stride0_,
                             int64_t size1_, int64_t stride1_,
                             int64_t size2_, int64_t stride2_,
                             int64_t size3_, int64_t stride3_)
{

  int64_t size[4] = {size0_, size1_, size2_, size3_};
  int64_t stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THCZTensor_(setStorageNd)(state, self, storage_, storageOffset_, 4, size, stride);
}


void THCZTensor_(narrow)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension, int64_t firstIndex, int64_t size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 4, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 5, "out of range");

  THCZTensor_(set)(state, self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THCZTensor_(select)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension, int64_t sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 4, "out of range");

  THCZTensor_(set)(state, self, src);
  THCZTensor_(narrow)(state, self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THCZTensor_(transpose)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension1, int dimension2)
{
  int64_t z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THCZTensor_(set)(state, self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THCZTensor_(unfold)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension, int64_t size, int64_t step)
{
  int64_t *newSize;
  int64_t *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THCZTensor_(set)(state, self, src);

  newSize = (int64_t*)THAlloc(sizeof(int64_t)*(self->nDimension+1));
  newStride = (int64_t*)THAlloc(sizeof(int64_t)*(self->nDimension+1));

  newSize[self->nDimension] = size;
  newStride[self->nDimension] = self->stride[dimension];
  for(d = 0; d < self->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (self->size[d] - size) / step + 1;
      newStride[d] = step*self->stride[d];
    }
    else
    {
      newSize[d] = self->size[d];
      newStride[d] = self->stride[d];
    }
  }

  THFree(self->size);
  THFree(self->stride);

  self->size = newSize;
  self->stride = newStride;
  self->nDimension++;
}

/* we have to handle the case where the result is a number */
void THCZTensor_(squeeze)(THCState *state, THCZTensor *self, THCZTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THCZTensor_(set)(state, self, src);

  for(d = 0; d < src->nDimension; d++)
  {
    if(src->size[d] != 1)
    {
      if(d != ndim)
      {
        self->size[ndim] = src->size[d];
        self->stride[ndim] = src->stride[d];
      }
      ndim++;
    }
  }

  /* right now, we do not handle 0-dimension tensors */
  if(ndim == 0 && src->nDimension > 0)
  {
    self->size[0] = 1;
    self->stride[0] = 1;
    ndim = 1;
  }
  self->nDimension = ndim;
}

void THCZTensor_(squeeze1d)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->nDimension, 3, "dimension out of range");

  THCZTensor_(set)(state, self, src);

  if(src->size[dimension] == 1 && src->nDimension > 1)
  {
    for(d = dimension; d < self->nDimension-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->nDimension--;
  }
}

void THCZTensor_(unsqueeze1d)(THCState *state, THCZTensor *self, THCZTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck((dimension >= 0) && (dimension <= src->nDimension), 3, "dimension out of range");
  THArgCheck(src->nDimension > 0, 3, "cannot unsqueeze empty tensor");

  THCZTensor_(set)(state, self, src);

  self->size = (int64_t*)THRealloc(self->size, sizeof(int64_t)*(self->nDimension+1));
  self->stride = (int64_t*)THRealloc(self->stride, sizeof(int64_t)*(self->nDimension+1));
  self->nDimension++;
  for (d = self->nDimension-1; d > dimension; d--) {
    self->size[d] = self->size[d-1];
    self->stride[d] = self->stride[d-1];
  }
  if (dimension+1 < self->nDimension) {
    self->stride[dimension] = self->size[dimension+1] * self->stride[dimension+1];
  } else {
    self->stride[dimension] = 1;
  }
  self->size[dimension] = 1;
}

int THCZTensor_(isContiguous)(THCState *state, const THCZTensor *self)
{
  int64_t z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}

int THCZTensor_(isSize)(THCState *state, const THCZTensor *self, const THLongStorage *dims)
{
  int d;
  if (self->nDimension != dims->size)
    return 0;

  for (d = 0; d < self->nDimension; ++d)
  {
    if (self->size[d] != dims->data[d])
      return 0;
  }
  return 1;
}

int THCZTensor_(isSetTo)(THCState *state, const THCZTensor *self, const THCZTensor *src)
{
  if (self->storage == src->storage &&
      self->storageOffset == src->storageOffset &&
      self->nDimension == src->nDimension)
  {
    int d;
    for (d = 0; d < self->nDimension; ++d)
    {
      if (self->size[d] != src->size[d] || self->stride[d] != src->stride[d])
        return 0;
    }
    return 1;
  }
  return 0;
}

int THCZTensor_(isSameSizeAs)(THCState *state, const THCZTensor *self, const THCZTensor* src)
{
  int d;
  if (self->nDimension != src->nDimension)
    return 0;
  for(d = 0; d < self->nDimension; ++d)
  {
    if(self->size[d] != src->size[d])
      return 0;
  }
  return 1;
}

ptrdiff_t THCZTensor_(nElement)(THCState *state, const THCZTensor *self)
{
  if(self->nDimension == 0)
    return 0;
  else
  {
    ptrdiff_t nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THCZTensor_(retain)(THCState *state, THCZTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    THAtomicIncrementRef(&self->refcount);
}

void THCZTensor_(free)(THCState *state, THCZTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(THAtomicDecrementRef(&self->refcount))
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THCZStorage_(free)(state, self->storage);
      THFree(self);
    }
  }
}

void THCZTensor_(freeCopyTo)(THCState *state, THCZTensor *self, THCZTensor *dst)
{
  if(self != dst)
    THCZTensor_(copy)(state, dst, self);

  THCZTensor_(free)(state, self);
}

/*******************************************************************************/

static void THCZTensor_(rawInit)(THCState *state, THCZTensor *self)
{
  self->refcount = 1;
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

void THCZTensor_(setStorageNd)(THCState *state, THCZTensor *self, THCZStorage *storage, ptrdiff_t storageOffset, int nDimension, int64_t *size, int64_t *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THCZStorage_(free)(state, self->storage);

    if(storage)
    {
      self->storage = storage;
      THCZStorage_(retain)(state, self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THCZTensor_(resizeNd)(state, self, nDimension, size, stride);
}

void THCZTensor_(resizeNd)(THCState *state, THCZTensor *self, int nDimension, int64_t *size, int64_t *stride)
{
  int d;
  int nDimension_;
  ptrdiff_t totalSize;
  int hascorrectsize = 1;

  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
    {
      nDimension_++;
      if((self->nDimension > d) && (size[d] != self->size[d]))
        hascorrectsize = 0;

      if((self->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hascorrectsize = 0;
    }
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension != self->nDimension)
    hascorrectsize = 0;

  if(hascorrectsize)
    return;

  if(nDimension > 0)
  {
    if(nDimension != self->nDimension)
    {
      self->size = (int64_t*)THRealloc(self->size, sizeof(int64_t)*nDimension);
      self->stride = (int64_t*)THRealloc(self->stride, sizeof(int64_t)*nDimension);
      self->nDimension = nDimension;
    }

    totalSize = 1;
    for(d = self->nDimension-1; d >= 0; d--)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == self->nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if(totalSize+self->storageOffset > 0)
    {
      if(!self->storage)
        self->storage = THCZStorage_(new)(state);
      if(totalSize+self->storageOffset > self->storage->size)
        THCZStorage_(resize)(state, self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THCZTensor_(set1d)(THCState *state, THCZTensor *tensor, int64_t x0, ntype value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THCZStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

ntype THCZTensor_(get1d)(THCState *state, const THCZTensor *tensor, int64_t x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THCZStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THCZTensor_(set2d)(THCState *state, THCZTensor *tensor, int64_t x0, int64_t x1, ntype value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THCZStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

ntype THCZTensor_(get2d)(THCState *state, const THCZTensor *tensor, int64_t x0, int64_t x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THCZStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THCZTensor_(set3d)(THCState *state, THCZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, ntype value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THCZStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

ntype THCZTensor_(get3d)(THCState *state, const THCZTensor *tensor, int64_t x0, int64_t x1, int64_t x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THCZStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THCZTensor_(set4d)(THCState *state, THCZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, ntype value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THCZStorage_(set)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

ntype THCZTensor_(get4d)(THCState *state, const THCZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return THCZStorage_(get)(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
}

int THCZTensor_(checkGPU)(THCState *state, unsigned int nTensors, ...)
{
  /* FIXME: remove this flag after any users stop using it since it is
     now superseded by the runtime option */
#ifdef DISABLE_CHECK_GPU
  return 1;
#else
  int kernelP2PEnabled =
    THCState_getKernelPeerToPeerAccessEnabled(state);

  int curDev = -1;
  THCudaCheck(cudaGetDevice(&curDev));
  va_list(args);
  va_start(args, nTensors);
  int valid = 1;
  for (unsigned int i = 0; i < nTensors; i++) {
    THCZTensor* tensor = va_arg(args, THCZTensor*);
    if (tensor == NULL) {
      continue;
    }
    int tensorDev = THCZTensor_(getDevice)(state, tensor);
    if (tensorDev == -1) {
      /* This tensor does not have GPU memory (empty) */
      continue;
    }

    if (tensorDev != curDev) {
      if (kernelP2PEnabled) {
        /* Kernel p2p access is allowed */
        /* Can `curDev` access `tensorDev` directly? */
        if (!THCState_getPeerToPeerAccess(state, curDev, tensorDev)) {
          valid = 0;
          break;
        }
      } else {
        /* No kernel p2p access allowed */
        valid = 0;
        break;
      }
    }
  }

  va_end(args);
  return valid;
#endif // DISABLE_CHECK_GPU
}

THCDescBuff THCZTensor_(sizeDesc)(THCState *state, const THCZTensor *tensor) {
  const int L = THC_DESC_BUFF_LEN;
  THCDescBuff buf;
  char *str = buf.str;
  int n = 0;
  n += snprintf(str, L-n, "[");
  int i;
  for(i = 0; i < tensor->nDimension; i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%" PRId64, tensor->size[i]);
    if(i < tensor->nDimension-1) {
      n += snprintf(str+n, L-n, " x ");
    }
  }
  if(n < L - 2) {
    snprintf(str+n, L-n, "]");
  } else {
    snprintf(str+L-5, 5, "...]");
  }
  return buf;
}

#endif
