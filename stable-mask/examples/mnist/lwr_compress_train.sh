#!/usr/bin/env sh

./build/tools/caffe train  \
    --solver=examples/mnist/lwr_compress_solver.prototxt  \
    --weights=/home/liuwr/liuwenran/compress/surgery/output/lenet5/rollback/rollback_scratch.caffemodel
