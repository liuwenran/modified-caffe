#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/double_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
DoubleDataLayer<Dtype>::DoubleDataLayer(const LayerParameter& param)
  : BiPrefetchDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DoubleDataLayer<Dtype>::~DoubleDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DoubleDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  this->second_size_ = this->layer_param_.double_data_param().second_size();
  this->second_whole_size_ = this->layer_param_.double_data_param().second_whole_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);

  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  vector<int> top_second_shape(top_shape);
  top_second_shape[2] = this->second_size_;
  top_second_shape[3] = this->second_size_;

  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_second_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
    this->prefetch_[i].secondData_.Reshape(top_second_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  LOG(INFO) << "output second data size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[2]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DoubleDataLayer<Dtype>::load_batch(DoubleBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  vector<int> top_second_shape(top_shape);
  top_second_shape[2] = this->second_size_;
  top_second_shape[3] = this->second_size_;

  vector<int> temp_shape(top_second_shape);
  temp_shape[0] = 1;
  this->transformed_secondData_.Reshape(temp_shape);

  batch->data_.Reshape(top_shape);
  batch->secondData_.Reshape(top_second_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_second_data = batch->secondData_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    int second_offset = batch->secondData_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->transformed_secondData_.set_cpu_data(top_second_data + second_offset);
    // this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // this->data_transformer_->TransformToSecond(&(this->transformed_data_), &(this->transformed_secondData_), this->second_size_);
    this->data_transformer_->TransformBoth(datum, &(this->transformed_data_), &(this->transformed_secondData_), this->second_whole_size_, this->second_size_);

    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DoubleDataLayer);
REGISTER_LAYER_CLASS(DoubleData);

}  // namespace caffe
