#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/imagenet/lwr_compress_solver.prototxt \
    --weights=/home/liuwr/liuwenran/compress/surgery/output/alexnet/incremental/fcv_lr0.001_noWD_iter_300000.caffemodel \
    --gpu 2
