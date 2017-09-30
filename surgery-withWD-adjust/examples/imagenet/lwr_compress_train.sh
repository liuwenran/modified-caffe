#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/imagenet/lwr_compress_solver.prototxt \
    --weights=/home/liuwr/liuwenran/compress/surgery/output/alexnet/withWD/prune_conv4_5_lr0.001_cate2_iter_300000.caffemodel \
    --gpu 3
