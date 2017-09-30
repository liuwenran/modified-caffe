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
void blobMaxMin(const int n, const Dtype* blob, Dtype * max, Dtype * min) {
  Dtype maxnow = blob[0];
  Dtype minnow = blob[0];

  for (int i = 1; i < n; i++) {
    if (blob[i] > maxnow) {
      maxnow = blob[i];
    }
    if (blob[i] < minnow) {
      minnow = blob[i];
    }
  }
  *max = maxnow;
  *min = minnow;
}

template <typename Dtype>
__global__ void CCMaskAdjust(const int n, Dtype* mask) {
  CUDA_KERNEL_LOOP(index, n) {
    // if (mask[index] > 0) {
    //   mask[index] = 1;
    // }
    // else {
    //   mask[index] = 0;
    // }
    mask[index] = int (mask[index] + 0.5);
  }
}

template <typename Dtype>
__global__ void absdata(const int n, Dtype* mask ) {
  CUDA_KERNEL_LOOP (index, n) {
    mask[index] = fabs(mask[index]);
  }
}

template <typename Dtype>
__global__ void classInit(const int n, Dtype* blobClass, const int numplusone){
  CUDA_KERNEL_LOOP (index, n) {
    blobClass[index] = numplusone;
  }
}

template <typename Dtype>
__global__ void CUassignClass(const int n, const Dtype* blob, Dtype* blobClass, const int m,
  Dtype * edge, const int negb, const int nege, const int posb, const int pose){
  CUDA_KERNEL_LOOP (index, n) {
    if ( int(blobClass[index]) == m -1)
    {
      for (int i = 0; i < m - 1; ++i)
      {
        if (blob[index] >= edge[i] && blob[index] <= edge[i+1])
        {
          if ( (i>= negb && i<=nege) || (i >= posb && i <= pose) )
          {
            blobClass[index] = i;
          }
        }
      }
    }

  }
}

template <typename Dtype>
void assignClass(const int n, const Dtype * blob, Dtype* blobClass, const int m, 
  Dtype * edge, const int negb, const int nege, const int posb, const int pose) {
  Dtype * cuedge;
  cudaMalloc(&cuedge, sizeof(Dtype) * m);
  cudaMemcpy(cuedge, edge, sizeof(Dtype) * m, cudaMemcpyHostToDevice);    
  CUassignClass<Dtype><<<CAFFE_GET_BLOCKS(n),
    CAFFE_CUDA_NUM_THREADS>>>(n, blob, blobClass, m, cuedge, negb, nege, posb, pose);
  CUDA_POST_KERNEL_CHECK;
  cudaFree(cuedge);

}

template <typename Dtype>
__global__ void CUassignValue(const int n, const Dtype * blob, Dtype* blobClass,
  Dtype * blobTmp, const int m,  Dtype * cucenter){
  CUDA_KERNEL_LOOP (index, n){
    if (blobClass[index] < m)
    {
      blobTmp[index] = cucenter[int(blobClass[index])];
    }
    else
    {
      blobTmp[index] = blob[index];
    }
  }
}


template <typename Dtype>
void assignValue(const int n, const Dtype * blob, Dtype* blobClass,
  Dtype * blobTmp, const int m,  Dtype * center){
  Dtype* cucenter;
  cudaMalloc(&cucenter, sizeof(Dtype) *m);
  cudaMemcpy(cucenter, center, sizeof(Dtype) * m, cudaMemcpyHostToDevice);
  CUassignValue<Dtype><<<CAFFE_GET_BLOCKS(n), 
    CAFFE_CUDA_NUM_THREADS>>>(n, blob, blobClass, blobTmp, m, cucenter);
  CUDA_POST_KERNEL_CHECK; 
  cudaFree(cucenter);
}

template <typename Dtype>
__global__ void CUclassToMask(const int n, const Dtype * blobClass, Dtype* blobMask, const int classnum){
  CUDA_KERNEL_LOOP (index, n) {
    if (blobClass[index] < classnum)
    {
      blobMask[index] = 0;
    }
    else{
      blobMask[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void CUmaskAllOne(const int n , Dtype * blob){
  CUDA_KERNEL_LOOP (index, n) {
    blob[index] = 1;
  }
}

template <typename Dtype>
void testCopy(const int m, Dtype * edge, Dtype * fromgpu) {
  Dtype * cuedge;
  cudaMalloc(&cuedge, sizeof(Dtype) * m);
  cudaMemcpy(cuedge, edge, sizeof(Dtype) * m, cudaMemcpyHostToDevice);    

  cudaMemcpy(fromgpu, cuedge, sizeof(Dtype) * m, cudaMemcpyDeviceToHost);
  cudaFree(cuedge);

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
void PQInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {    

  const Dtype* weight = this->blobs_[0]->mutable_gpu_data();  
  Dtype* weightClass = this->blobs_[2]->mutable_gpu_data();
  Dtype* weightTmp = this->weight_tmp_.mutable_gpu_data(); 
  const Dtype* bias = NULL;
  Dtype* biasClass = NULL;
  Dtype* biasTmp = NULL;   
  if (this->bias_term_) {  
    bias = this->blobs_[1]->mutable_gpu_data();  
    biasClass = this->blobs_[3]->mutable_gpu_data(); 
    biasTmp = this->bias_tmp_.mutable_gpu_data();
  }



  if (pow(2,this->qstage_) > this->classnum_)
  {
    LOG(ERROR)<<"bad qstage"<<"\n";
  }
  
  if (this->phase_ == TRAIN && this->iter_ == 0 && this->qstage_ > 0 && this->classnum_ > 0  ){


      if (this->qstage_ == 1)
      {
        classInit<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
          CAFFE_CUDA_NUM_THREADS>>>(this->blobs_[0]->count(), 
            this->blobs_[2]->mutable_gpu_data(), this->classnum_);
        CUDA_POST_KERNEL_CHECK; 
        classInit<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
          CAFFE_CUDA_NUM_THREADS>>>(this->blobs_[1]->count(), 
            this->blobs_[3]->mutable_gpu_data(), this->classnum_);
        CUDA_POST_KERNEL_CHECK;
      }

      std::cout<<"afterInit:"<<std::endl;
      // const Dtype* afterInit = wclassCopy.cpu_data();
      const Dtype* afterInit = this->blobs_[2]->cpu_data();
      for (int i = 0; i < 100; ++i)
      {
        std::cout<<i<<":"<<afterInit[i]<<" ";
        // std::cout<<i<<":"<<beforeAssign[i]<<" ";
      }
      std::cout<<std::endl;

      Blob<Dtype> wsort(this->blobs_[0]->shape());
      Blob<Dtype> bsort(this->blobs_[1]->shape());
      wsort.CopyFrom(*(this->blobs_[0]));
      bsort.CopyFrom(*(this->blobs_[1]));


      
      absdata<Dtype><<<CAFFE_GET_BLOCKS(wsort.count()),CAFFE_CUDA_NUM_THREADS>>>(
         wsort.count(), wsort.mutable_gpu_data());
      CUDA_POST_KERNEL_CHECK;
      absdata<Dtype><<<CAFFE_GET_BLOCKS(bsort.count()),CAFFE_CUDA_NUM_THREADS>>>(
         bsort.count(), bsort.mutable_gpu_data());
      CUDA_POST_KERNEL_CHECK; 



      const Dtype* cpuWeightClass = this->blobs_[2]->cpu_data();
      const Dtype* cpuBiasClass = this->blobs_[3]->cpu_data();

      int paramnums = wsort.count() + bsort.count();

      Dtype* allparams = (Dtype*) malloc(paramnums * sizeof(Dtype));

      int temcount = 0;
      for (int i = 0; i < wsort.count(); ++i)
      {
        if (cpuWeightClass[i] == this->classnum_)
        {
          allparams[temcount] = wsort.mutable_cpu_data()[i];
          temcount++;
        }
      }
      for (int i = 0; i < bsort.count(); ++i)
      {
        if (cpuBiasClass[i] == this->classnum_)
        {
          allparams[temcount] = bsort.mutable_cpu_data()[i];
          temcount++;
        }
      }
      for (int i = temcount; i < paramnums; ++i)
      {
        allparams[i] = 0;
      }



      Dtype absmax = 0;
      Dtype absmin = 0;
      blobMaxMin(temcount, allparams, &absmax, &absmin);

      std::cout<<"temcount:"<<temcount<<std::endl;

      Dtype median = findMedian(allparams, 0, temcount - 1, temcount/2);
      free(allparams);

      // std::cout<<"please input a number:"<<std::endl;
      // int input;
      // std::cin>>input;

      int midclass = this->classnum_ / 2;
      int maxstage = log(this->classnum_) / log(2);
      int nowclassnum = midclass / pow(2, this->qstage_);
      if (nowclassnum == 0)
      {
        nowclassnum = 1;
      }

      int negb = midclass - pow(2, maxstage - this->qstage_);
      int nege = negb + nowclassnum - 1;
      int pose = midclass + pow(2, maxstage - this->qstage_) - 1;
      int posb = pose - nowclassnum + 1;

      Dtype * edge = (Dtype*) malloc((this->classnum_ + 1) * sizeof(Dtype));
      for (int i = 0; i < this->classnum_ + 1; ++i)
      {
        edge[i] = 0;
      }
      

      float internal = (absmax - median) / nowclassnum;
      if (this->qstage_ == maxstage)
      {
        internal = absmax;
      }
      for (int i = 0; i < nowclassnum + 1; ++i)
      {
        edge[negb + i] = -absmax + internal * i;
        edge[pose + 1 - i] = absmax - internal * i; 
      }

      std::cout<<"edge:"<<std::endl;
      for (int i = 0; i < this->classnum_ + 1; ++i)
      {
        std::cout<<edge[i]<<" ";
      }
      std::cout<<std::endl;


      assignClass(this->blobs_[0]->count(), weight, this->blobs_[2]->mutable_gpu_data(),
        this->classnum_ + 1, edge, negb, nege, posb, pose);
      assignClass(this->blobs_[1]->count(), bias, this->blobs_[3]->mutable_gpu_data(),
        this->classnum_ + 1, edge, negb, nege, posb, pose);


      Dtype * weightMask = this->weight_mask_.mutable_gpu_data();
      Dtype * biasMask = this->bias_mask_.mutable_gpu_data();
      CUclassToMask<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
        CAFFE_CUDA_NUM_THREADS>>>(this->blobs_[0]->count(), 
          this->blobs_[2]->mutable_gpu_data(), weightMask, this->classnum_);
      CUDA_POST_KERNEL_CHECK;
      CUclassToMask<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
        CAFFE_CUDA_NUM_THREADS>>>(this->blobs_[1]->count(), 
          this->blobs_[3]->mutable_gpu_data(), biasMask, this->classnum_);
      CUDA_POST_KERNEL_CHECK;


      const Dtype* wsumClass = this->blobs_[2]->cpu_data();
      const Dtype* wsum = this->blobs_[0]->cpu_data();
      for (int i = 0; i < this->blobs_[2]->count(); ++i)
      {
        this->eachsum_[wsumClass[i]] += wsum[i];
        this->eachnum_[wsumClass[i]]++;
      }



      const Dtype* bsumClass = this->blobs_[3]->cpu_data();
      const Dtype* bsum = this->blobs_[1]->cpu_data();


      for (int i = 0; i < this->blobs_[3]->count(); ++i)
      {
        this->eachsum_[bsumClass[i]] += bsum[i];
        this->eachnum_[bsumClass[i]]++;
      }

      std::cout<<"center before:"<<std::endl;
      for (int i = 0; i < this->classnum_ + 1; ++i)
      {
        std::cout<<this->center_[i]<<" ";
      }
      std::cout<<std::endl;

      std::cout<<"eachnum:"<<std::endl;
      for (int i = 0; i < this->classnum_ + 1; ++i)
      {
        std::cout<<this->eachnum_[i]<<" ";
      }
      std::cout<<std::endl;

      // if (false)
      // {
      // // // testCopy(this->classnum_+ 1, edge, fromgpu);
      std::cout<<"center:"<<std::endl;
      for (int i = 0; i < this->classnum_ + 1; ++i)
      {
        if (this->eachnum_[i] > 0)
        {
          this->center_[i] = this->eachsum_[i] / this->eachnum_[i];
        }
        std::cout<<center_[i]<<" ";
      }
      std::cout<<std::endl;


      std::cout<<"original weight:"<<std::endl;
      const Dtype* checkOrigin = this->blobs_[0]->cpu_data();
      for (int i = 0; i < 100; ++i)
      {
        std::cout<<i<<":"<<checkOrigin[i]<<" ";
      }
      std::cout<<std::endl;

      std::cout<<"weigth class:"<<std::endl;
      const Dtype* checkClass = this->blobs_[2]->cpu_data();
      for (int i = 0; i < 100; ++i)
      {
        std::cout<<i<<":"<<checkClass[i]<<" ";
      }
      std::cout<<std::endl;

      std::cout <<"paramscount:"<< paramnums<<std::endl;
      std::cout<<"negb:"<<negb<<std::endl;
      std::cout<<"nege:"<<nege<<std::endl;
      std::cout<<"posb:"<<posb<<std::endl;
      std::cout<<"pose:"<<pose<<std::endl;
      std::cout<<"nowclassnum:"<<nowclassnum<<std::endl;
      std::cout<<"absmax:"<<absmax<<std::endl;
      std::cout<<"median:"<<median<<std::endl;
      std::cout<<"qstage:"<<this->qstage_<<std::endl;


      Dtype * setConfig = this->blobs_[4]->mutable_cpu_data();
      setConfig[0] = 1;
      for (int i = 1; i < this->classnum_ + 2; ++i)
      {
        setConfig[i] = this->center_[i-1];
      }
   
  }   

  const Dtype* getConfig = this->blobs_[4]->cpu_data();
  if (getConfig[0] == 0)
  {
    Dtype * weightMask = this->weight_mask_.mutable_gpu_data();
    Dtype * biasMask  = this->bias_mask_.mutable_gpu_data();
    CUmaskAllOne<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
      CAFFE_CUDA_NUM_THREADS>>>(this->blobs_[0]->count(), weightMask);
    CUDA_POST_KERNEL_CHECK;
    CUmaskAllOne<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
      CAFFE_CUDA_NUM_THREADS>>>(this->blobs_[1]->count(), biasMask);
    CUDA_POST_KERNEL_CHECK;
    
    CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[0]->count()),
     CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[0]->count(), weight, weightMask, weightTmp);
    CUDA_POST_KERNEL_CHECK;
    CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[1]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[1]->count(), bias, biasMask, biasTmp);
    CUDA_POST_KERNEL_CHECK;  
  }
  else
  {
    for (int i = 1; i < this->classnum_ + 2; ++i)
    {
      this->center_[i-1] = getConfig[i];
    }
    assignValue(this->blobs_[0]->count(), weight, weightClass, weightTmp, 
     this->classnum_ , this->center_);
    assignValue(this->blobs_[1]->count(), bias, biasClass, biasTmp, 
      this->classnum_ , this->center_);    
  }


  if (this->iter_ == 0)
  {
    std::cout<<"weight temp:"<<std::endl;
    Dtype* checktmp = this->weight_tmp_.mutable_cpu_data();
    for (int i = 0; i < 100; ++i)
    {
      std::cout<<i<<":"<<checktmp[i]<<" ";
    }
    std::cout<<std::endl;
  }
   
	// Forward calculation with (masked) weight and bias 
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weightTmp, bottom_data, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            biasTmp, top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weightTmp, (Dtype)0., top_data);
    if (this->bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            biasTmp, (Dtype)1., top_data);
  }

  this->iter_++;
}

template <typename Dtype>
void PQInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
		// const Dtype* weightMask = this->blobs_[2]->gpu_data();
    const Dtype* weightMask = this->weight_mask_.gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
		CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[2]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[2]->count(), weight_diff, weightMask, weight_diff);
    CUDA_POST_KERNEL_CHECK; 
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., weight_diff);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
		// const Dtype* biasMask = this->blobs_[3]->gpu_data();
    const Dtype* biasMask = this->bias_mask_.gpu_data();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    // Gradient with respect to bias
    CCMaskApply<Dtype><<<CAFFE_GET_BLOCKS(this->blobs_[3]->count()),
      CAFFE_CUDA_NUM_THREADS>>>( this->blobs_[3]->count(), bias_diff, biasMask, bias_diff);
    CUDA_POST_KERNEL_CHECK; 		
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,bias_diff);
  }	
  if (propagate_down[0]) {
		const Dtype* weightTmp = this->weight_tmp_.gpu_data();        
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weightTmp, (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PQInnerProductLayer);

}  // namespace caffe
