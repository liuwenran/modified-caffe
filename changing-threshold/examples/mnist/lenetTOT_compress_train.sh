#!/usr/bin/env sh

./build/tools/caffe train  \
    --solver=examples/mnist/lenetTOT_compress_solver.prototxt  \
    --weights=/home/liuwr/liuwenran/compress/distilling/output/lenetTOT/lenetTOT_compress_0.5_0.5_0.9_0.5_0.5_0.1_iter_20000.caffemodel
