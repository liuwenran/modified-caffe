#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/imagenet/pqCompress_all_stage1_solver.prototxt \
    --weights=/home/liuwr/liuwenran/compress/pqCompress/output/alexnet/alexnet_pqCompress_all_reference_borrowed.caffemodel \
    --gpu 1
