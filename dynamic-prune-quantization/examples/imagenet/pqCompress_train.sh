#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/imagenet/pqCompress_solver.prototxt \
    --weights=/net/liuwenran/compress/pqCompress/output/alexnet/alexnet_pqCompress_reference_borrowed.caffemodel \
    --gpu 0
