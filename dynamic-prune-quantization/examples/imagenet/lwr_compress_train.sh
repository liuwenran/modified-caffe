#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/imagenet/lwr_compress_solver.prototxt \
    --weights=/net/liuwenran/compress/surgery/output/alexnet/allChanged/alexnet_compress_reference_borrowed.caffemodel \
    --gpu 2
