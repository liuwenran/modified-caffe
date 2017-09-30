#!/usr/bin/env sh

./build/tools/caffe train  \
    --solver=examples/mnist/lwr_compress_solver.prototxt  \
    --weights=/home/liuwr/liuwenran/compress/surgery/output/lenet5/allChanged/lenet_compress_all_reference_iter_10000.caffemodel
