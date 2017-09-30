#include <vector>

//#include "caffe/common_layers.hpp"
//#include "caffe/layer.hpp"
#include "caffe/layers/eliminate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EliminateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->mutable_cpu_data()[0] = 666;
}

template <typename Dtype>
void EliminateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(EliminateLayer);

}  // namespace caffe