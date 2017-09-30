#!/usr/bin/env sh

./build/tools/caffe test --model=models/bvlc_alexnet/train_val.prototxt --weights=models/bvlc_alexnet/bvlc_alexnet.caffemodel  --iterations=1000 --gpu 3 
