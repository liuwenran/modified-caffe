#!/usr/bin/env sh

./build/tools/caffe train  \
   --solver=examples/mnist/pqcompress_all_stage0_solver.prototxt  \
#   --weights=/net/liuwenran/compress/pqCompress/output/lenet5/before_conv_qstage1_iter_10000.caffemodel 
