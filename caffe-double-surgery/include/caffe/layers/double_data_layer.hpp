#ifndef CAFFE_DOUBLE_DATA_LAYERS_HPP_
#define CAFFE_DOUBLE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

#include "caffe/data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
class DoubleBatch {
 public:
  Blob<Dtype> data_, secondData_, label_;
};

template <typename Dtype>
class BiPrefetchDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BiPrefetchDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(DoubleBatch<Dtype>* batch) = 0;

  DoubleBatch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<DoubleBatch<Dtype>*> prefetch_free_;
  BlockingQueue<DoubleBatch<Dtype>*> prefetch_full_;

  Blob<Dtype> transformed_data_,transformed_secondData_;

  int second_size_;
  int second_whole_size_;
};

template <typename Dtype>
class DoubleDataLayer : public BiPrefetchDataLayer<Dtype> {
 public:
  explicit DoubleDataLayer(const LayerParameter& param);
  virtual ~DoubleDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "DoubleData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  virtual void load_batch(DoubleBatch<Dtype>* batch);

  DataReader reader_;
};



}  // namespace caffe

#endif  // CAFFE_DOUBLE_DATA_LAYERS_HPP_
