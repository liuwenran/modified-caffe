#!/usr/bin/env sh

./build/tools/caffe test --model=models/bvlc_alexnet/train_val.prototxt --weights=/net/liuwenran/compress/Deep-Compression-AlexNet/decompress_alexnet.caffemodel  --iterations=1000 --gpu 3 
