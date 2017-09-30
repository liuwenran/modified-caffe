#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <cmath>

namespace caffe {

// The constant NUM_THREADS should be equal to the value in CCMomentCalc
template <typename Dtype>
__global__ void CCMomentCollect(const int n, const Dtype* wb, const Dtype* mask,
    Dtype* mu, Dtype* std, unsigned int* count ) {  
  const int NUM_THREADS = 512;  
  __shared__ Dtype param [4*NUM_THREADS]; 
  __shared__ unsigned int tcount [2*NUM_THREADS];   
  unsigned int t = threadIdx.x; 
  unsigned int s = 2 * blockIdx.x * NUM_THREADS;
  if (s+t < n){
    param[t] = fabs(mask[s+t]*wb[s+t]);
    param[t+2*NUM_THREADS] = mask[s+t]*wb[s+t]*wb[s+t];
    if(mask[s+t]*wb[s+t]!=0) tcount[t] = 1;
    else tcount[t] = 0;
  }
  else{
    param[t] = 0;param[t+2*NUM_THREADS] = 0;tcount[t] = 0;
  }
  if (s+t+NUM_THREADS < n){
    param[t+NUM_THREADS] = fabs(mask[s+t+NUM_THREADS]*wb[s+t+NUM_THREADS]);
    param[t+3*NUM_THREADS] = mask[s+t+NUM_THREADS]*wb[s+t+NUM_THREADS]*wb[s+t+NUM_THREADS];
    if(mask[s+t+NUM_THREADS]*wb[s+t+NUM_THREADS]!=0) tcount[t+NUM_THREADS] = 1;
    else tcount[t+NUM_THREADS] = 0;
  }
  else{
    param[t+NUM_THREADS] = 0;param[t+3*NUM_THREADS] = 0;tcount[t+NUM_THREADS] = 0;  
  }
  __syncthreads(); 
  for(unsigned int stride = NUM_THREADS; stride >= 1; stride >>= 1) {
    if (t < stride ){
      param[t] += param[t+stride]; 
      param[t+2*NUM_THREADS] += param[t+2*NUM_THREADS+stride];
      tcount[t] += tcount[t+stride];
    }
    __syncthreads();  
  }
  if (t == 0){
    mu   [blockIdx.x] = param[0];
    std  [blockIdx.x] = param[2*NUM_THREADS];
    count[blockIdx.x] = tcount[0]; 
  }      
}

// The constant NUM_THREADS should be equal to the value in CCMomentCalc
template <typename Dtype>
__global__ void CCNzeroCollect(const int n, const Dtype* mask, unsigned int* count ) {  
  const int NUM_THREADS = 512;  
  __shared__ unsigned int tcount [2*NUM_THREADS];   
  unsigned int t = threadIdx.x; 
  unsigned int s = 2 * blockIdx.x * NUM_THREADS;
  tcount[t] = 0;
  if (s+t < n && mask[s+t]!=0){
    tcount[t] = 1;
  }
  tcount[t+NUM_THREADS] = 0;
  if (s+t+NUM_THREADS < n && mask[s+t+NUM_THREADS]!=0){
    tcount[t+NUM_THREADS] = 1;
  }
  __syncthreads(); 
  for(unsigned int stride = NUM_THREADS; stride >= 1; stride >>= 1) {
    if (t < stride ){
      tcount[t] += tcount[t+stride];
    }
    __syncthreads();  
  }
  if (t == 0){
    count[blockIdx.x] = tcount[0]; 
  }     
}

template <typename Dtype>
__global__ void CCMaskCalc(const int n, const Dtype* wb,
    Dtype* mask, Dtype mu, Dtype std, Dtype r) {
  CUDA_KERNEL_LOOP(index, n) {
    if (mask[index]==1 && fabs(wb[index])<=0.9*max(mu+r*std,Dtype(0))) 
      mask[index] = 0;
    else if (mask[index]==0 && fabs(wb[index])>1.1*max(mu+r*std,Dtype(0)))
      mask[index] = 1;
  }
}

template <typename Dtype>
__global__ void CCMaskApply(const int n, const Dtype* wb,
    const Dtype* mask, Dtype* wb_t) {
  CUDA_KERNEL_LOOP(index, n) {
    wb_t[index] = wb[index] * mask[index];    
  }
}

template <typename Dtype>
void CCMomentCalc(const int n, const Dtype* wb, const Dtype* mask, Dtype* mu, Dtype* std, unsigned int* ncount){ 
  const unsigned int NUM_THREADS = 512;
  Dtype* pmu_g; Dtype* pstd_g; unsigned int* pncount_g;
  Dtype* pmu_c; Dtype* pstd_c; unsigned int* pncount_c;
  int num_p = (n+(NUM_THREADS<<1)-1)/(NUM_THREADS<<1);  
  cudaMalloc(&pmu_g, sizeof(Dtype)  * num_p);
  cudaMalloc(&pstd_g, sizeof(Dtype) * num_p);
  cudaMalloc(&pncount_g, sizeof(unsigned int) * num_p);
  pmu_c = (Dtype*) malloc(num_p * sizeof(Dtype));
  pstd_c = (Dtype*) malloc(num_p * sizeof(Dtype)); 
  pncount_c = (unsigned int*) malloc(num_p * sizeof(unsigned int));      
  CCMomentCollect<Dtype><<<num_p,NUM_THREADS>>>(n, wb, mask, pmu_g, pstd_g, pncount_g);
  CUDA_POST_KERNEL_CHECK; 
  cudaMemcpy(pmu_c, pmu_g, sizeof(Dtype) * num_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(pstd_c, pstd_g, sizeof(Dtype) * num_p, cudaMemcpyDeviceToHost);
  cudaMemcpy(pncount_c, pncount_g, sizeof(unsigned int) * num_p, cudaMemcpyDeviceToHost);      
  for (int i = 0; i < num_p; i++) {
    *mu += pmu_c[i];*std += pstd_c[i];*ncount += pncount_c[i];
  }       
  cudaFree(pmu_g);cudaFree(pstd_g);cudaFree(pncount_g);
  free(pmu_c);free(pstd_c);free(pncount_c);
}

template <typename Dtype>
void CCNZeroCalc(const int n, const Dtype* mask, unsigned int* ncount ){  
  const unsigned int NUM_THREADS = 512;
  unsigned int* pncount_g;
  unsigned int* pncount_c;
  int num_p = (n+(NUM_THREADS<<1)-1)/(NUM_THREADS<<1);  
  cudaMalloc(&pncount_g, sizeof(unsigned int) * num_p);
  pncount_c = (unsigned int*) malloc(num_p * sizeof(unsigned int));      
  CCNzeroCollect<Dtype><<<num_p,NUM_THREADS>>>(n, mask, pncount_g);
  CUDA_POST_KERNEL_CHECK; 
  cudaMemcpy(pncount_c, pncount_g, sizeof(unsigned int) * num_p, cudaMemcpyDeviceToHost);      
  for (int i = 0; i < num_p; i++) {
    *ncount += pncount_c[i];
  }       
  cudaFree(pncount_g);
  free(pncount_c);
}

template <typename Dtype>
__global__ void absdata(const int n, Dtype* mask ) {
  CUDA_KERNEL_LOOP (index, n) {
    mask[index] = fabs(mask[index]);
  }
}

template <typename Dtype>
__global__ void CUnewMaskCalc(const int n, const Dtype* wb,
    Dtype* mask, Dtype cutLeft, Dtype cutRight) {
  CUDA_KERNEL_LOOP(index, n) {
    if (mask[index]==1 && fabs(wb[index])<= max(cutLeft,Dtype(0))) 
      mask[index] = 0;
    else if (mask[index]==0 && fabs(wb[index])> max(cutRight,Dtype(0)))
      mask[index] = 1;
  }
}


template <typename Dtype>
__global__ void CUmaskCombine(const int n, const Dtype* lastMask, Dtype* mask) {
  CUDA_KERNEL_LOOP(index, n) {
    if (lastMask[index]== 0 ) 
      mask[index] = 0;
  }
}

template <typename Dtype>
int partition(Dtype * data, int p ,int r) {
  Dtype x = data[r];
  Dtype temp;
  int i = p - 1;
  for(int j = p; j<r; j++) {
    if (data[j] <= x)
    {
      i = i + 1;
      temp = data[i];
      data[i] = data[j];
      data[j] = temp;
    }
  }
  temp = data[i+1];
  data[i+1] = data[r];
  data[r] = temp;
  return i+1;
}

template <typename Dtype>
Dtype findMedian(Dtype * data, int p, int r, int i){
  if (p == r)
  {
    return data[p];
  }
  int q = partition(data, p, r);
  int k = q - p + 1;
  if (i == k)
  {
    return data[q];
  }
  else if (i < k)
  {
    return findMedian(data, p, q - 1, i);
  }
  else
  {
    return findMedian(data, q+1, r, i-k);
  }
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  

  const Dtype* weight = this->blobs_[0]->mutable_gpu_data();  
  Dtype* weightMask = this->blobs_[2]->mutable_gpu_data();
  Dtype* weightTmp = this->weight_tmp_.mutable_gpu_data(); 
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;
  Dtype* biasTmp = NULL;   
  if (this->bias_term_) {  
    bias = this->blobs_[1]->mutable_gpu_data();   
    biasMask = this->blobs_[3]->mutable_gpu_data();
    biasTmp = this->bias_tmp_.mutable_gpu_data();
  }
  
  if (this->phase_ == TRAIN){
		// Calculate the mean and standard deviation of learnable parameters 		
   



		
		// Demonstrate the sparsity of compressed convolutional layer
		/********************************************************/
    if(this->iter_%100==0){
      unsigned int wNoneZero = 0;
      unsigned int wAll = 0;
      unsigned int bNoneZero = 0;
      unsigned int bAll = 0;
      CCNZeroCalc(this->blobs_[0]->count(), weightMask, &wNoneZero);
      wAll = this->blobs_[0]->count();
      if (this->bias_term_) {  
        CCNZeroCalc(this->blobs_[1]->count(), biasMask, &bNoneZero); 
        bAll = this->blobs_[1]->count();  
      }
      //LOG(INFO)<<ncount<<"\n";        
      LOG(INFO)<<wNoneZero<<"  "<<wAll<<"  "<<bNoneZero<<"  "<<bAll<<"\n"; 
    } 
		/********************************************************/
		
		// Calculate the weight mask and bias mask with probability

  }   
 
  // Calculate the current (masked) weight and bias
  CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
    CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, weightTmp);
  CUDA_POST_KERNEL_CHECK;
  if (this->bias_term_) {  
    CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, biasMask, biasTmp);
    CUDA_POST_KERNEL_CHECK;  
  }
      
	// Forward calculation with (masked) weight and bias 
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weightTmp,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        this->forward_gpu_bias(top_data + top[i]->offset(n), biasTmp);
      }
    }
  }

  this->iter_++;
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weightTmp = this->weight_tmp_.gpu_data();  	
	const Dtype* weightMask = this->blobs_[2]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();  	
  for (int i = 0; i < top.size(); ++i) {    
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
			const Dtype* biasMask = this->blobs_[3]->gpu_data();
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();			
      CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[3]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[3]->count(), bias_diff, biasMask, bias_diff);
      CUDA_POST_KERNEL_CHECK;  
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[2]->count()),
        CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[2]->count(), weight_diff, weightMask, weight_diff);
      CUDA_POST_KERNEL_CHECK; 			
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + top[i]->offset(n), weightTmp,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CConvolutionLayer);

}  // namespace caffe
