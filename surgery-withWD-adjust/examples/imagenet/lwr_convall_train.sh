#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/imagenet/lwr_convall_solver.prototxt \
    --weights=/home/liuwr/liuwenran/compress/surgery/output/alexnet/allChanged/alexnet_compress_reference_borrowed.caffemodel \
    --gpu 2
