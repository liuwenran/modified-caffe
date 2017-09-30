#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <cmath>

namespace caffe {

template <typename Dtype>
void PQInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (this->bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  
  /************ For dynamic network surgery ***************/
	PQInnerProductParameter pqinner_param = this->layer_param_.pqinner_product_param();
	
  if(this->blobs_.size()==2 && (this->bias_term_)){
    this->blobs_.resize(5);
    // Intialize and fill the weightmask & biasmask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    caffe_set(this->blobs_[2]->count() , Dtype(pqinner_param.classnum()), 
      this->blobs_[2]->mutable_cpu_data());

    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    caffe_set(this->blobs_[3]->count() , Dtype(pqinner_param.classnum()), 
      this->blobs_[3]->mutable_cpu_data());
    
  }  
  else if(this->blobs_.size()==1 && (!this->bias_term_)){
    this->blobs_.resize(2);	  
    // Intialize and fill the weightmask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    caffe_set(this->blobs_[1]->count() , Dtype(pqinner_param.classnum()), 
      this->blobs_[1]->mutable_cpu_data());
   
  }   

  std::vector<int> configShape;
  configShape.resize(1);
  configShape[0] = 2048;
  this->blobs_[4].reset(new Blob<Dtype>(configShape));
  caffe_set(this->blobs_[4]->count() , Dtype(0), 
    this->blobs_[4]->mutable_cpu_data());

  
  // Intializing the tmp tensor
  this->weight_tmp_.Reshape(this->blobs_[0]->shape());
  this->bias_tmp_.Reshape(this->blobs_[1]->shape());  

  this->weight_mask_.Reshape(this->blobs_[0]->shape());
  this->bias_mask_.Reshape(this->blobs_[1]->shape());
  // Intialize the hyper-parameters
  this->std = 0;this->mu = 0;   
  this->gamma = pqinner_param.gamma(); 
  this->power = pqinner_param.power();
  this->crate = pqinner_param.c_rate();  
  this->iter_stop_ = pqinner_param.iter_stop();
  this->iter_ = 0;

  this->zeroThreshold_ = pqinner_param.zeroth();
  this->classnum_ = pqinner_param.classnum();
  this->qstage_ = pqinner_param.qstage();
  this->eachsum_.resize( int (this->classnum_) + 1);
  this->eachnum_.resize( int (this->classnum_) + 1);
  this->center_ = new Dtype[int (this->classnum_ ) + 1];
  for (int i = 0; i < this->classnum_ + 1; ++i)
  {
    this->eachnum_[i] = 0;
    this->eachsum_[i] = 0;
    this->center_[i] = 0;
  }
}

template <typename Dtype>
void PQInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void PQInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void PQInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {  
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(PQInnerProductLayer);
#endif

INSTANTIATE_CLASS(PQInnerProductLayer);
REGISTER_LAYER_CLASS(PQInnerProduct);

}  // namespace caffe
