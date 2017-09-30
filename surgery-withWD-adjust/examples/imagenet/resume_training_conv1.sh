#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/imagenet/lwr_compress_solver.prototxt \
    --snapshot=/net/liuwenran/compress/surgery/output/alexnet/withWD/prune_conv1_iter_150000.solverstate \
    --gpu 2
