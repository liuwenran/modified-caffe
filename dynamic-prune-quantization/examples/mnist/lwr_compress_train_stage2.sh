#!/usr/bin/env sh

./build/tools/caffe train  \
   --solver=examples/mnist/lwr_compress_solver_stage2.prototxt  \
   --weights=/net/liuwenran/compress/pqCompress/output/lenet5/conv_qstage1_iter_10000.caffemodel
