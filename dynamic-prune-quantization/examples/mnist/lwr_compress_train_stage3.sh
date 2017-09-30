#!/usr/bin/env sh

./build/tools/caffe train  \
   --solver=examples/mnist/lwr_compress_solver_stage3.prototxt  \
   --weights=/net/liuwenran/compress/pqCompress/output/lenet5/conv_qstage2_iter_501.caffemodel
