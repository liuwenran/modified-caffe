#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <cmath>

namespace caffe {

template <typename Dtype>
void PQConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer <Dtype>::LayerSetUp(bottom, top); 
  
  /************ For dynamic prune quantization ***************/
	PQConvolutionParameter pqconv_param = this->layer_param_.pqconvolution_param();
	
  if(this->blobs_.size()==2 && (this->bias_term_)){
    this->blobs_.resize(5);
    // Intialize and fill the weightmask & biasmask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    caffe_set(this->blobs_[2]->count() , Dtype(pqconv_param.classnum()), 
      this->blobs_[2]->mutable_cpu_data());

    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    caffe_set(this->blobs_[3]->count() , Dtype(pqconv_param.classnum()), 
      this->blobs_[3]->mutable_cpu_data());
   
  }  
  else if(this->blobs_.size()==1 && (!this->bias_term_)){
    this->blobs_.resize(2);	  
    // Intialize and fill the weightmask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    caffe_set(this->blobs_[1]->count() , Dtype(pqconv_param.classnum()), 
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
  this->gamma = pqconv_param.gamma(); 
  this->power = pqconv_param.power();
  this->crate = pqconv_param.c_rate();  
  this->iter_stop_ = pqconv_param.iter_stop();
  this->iter_ = 0;

  this->zeroThreshold_ = pqconv_param.zeroth();
  this->classnum_ = pqconv_param.classnum();
  this->qstage_ = pqconv_param.qstage();
  this->eachsum_.resize( int (this->classnum_) + 1);
  this->eachnum_.resize( int (this->classnum_) + 1);
  this->center_ = new Dtype[int (this->classnum_ ) + 1];
  for (int i = 0; i < this->classnum_ + 1; ++i)
  {
    this->eachnum_[i] = 0;
    this->eachsum_[i] = 0;
    this->center_[i] = 0;
  }
  /********************************************************/
}

template <typename Dtype>
void PQConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void PQConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
      NOT_IMPLEMENTED;
}

template <typename Dtype>
void PQConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(PQConvolutionLayer);
#endif

INSTANTIATE_CLASS(PQConvolutionLayer);
REGISTER_LAYER_CLASS(PQConvolution);

}  // namespace caffe
