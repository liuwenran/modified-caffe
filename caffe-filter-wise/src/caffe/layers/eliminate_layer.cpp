#include <vector>

#include "caffe/layers/eliminate_layer.hpp"

namespace caffe {

template <typename Dtype>
void EliminateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void EliminateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const vector<int> bottomshape = bottom[0]->shape();
  // vector<int> top_shape;
  // top_shape.push_back(1);
  // int rest = 1;
  // for(int i = 0;i < bottomshape.size();i ++)
  // {
  //   rest = bottomshape[i] * rest;
  // }
  // top_shape.push_back(rest);
  // top_shape.push_back(1);
  // top_shape.push_back(1);
  // top[0]->Reshape(top_shape);
  // CHECK_EQ(top[0]->count(), bottom[0]->count())
  //     << "output count must match input count";

  vector<int> top_shape;
  for(int i = 0;i < 4;i++)
  {
    top_shape.push_back(1);
  }
  top[0]->Reshape(top_shape);
  top[0]->mutable_cpu_data()[0] = 666;
}

INSTANTIATE_CLASS(EliminateLayer);
REGISTER_LAYER_CLASS(Eliminate);

}  // namespace caffe
