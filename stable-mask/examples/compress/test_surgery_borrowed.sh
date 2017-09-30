#!/usr/bin/env sh

./build/tools/caffe test --model=examples/imagenet/lwr_compress_train_val.prototxt --weights=/net/liuwenran/compress/surgery/output/alexnet/allChanged/alexnet_compress_reference_borrowed.caffemodel  --iterations=1000 --gpu 3 
