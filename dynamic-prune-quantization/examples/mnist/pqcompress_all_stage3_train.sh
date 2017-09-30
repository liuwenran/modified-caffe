#!/usr/bin/env sh

./build/tools/caffe train  \
   --solver=examples/mnist/pqcompress_all_stage3_solver.prototxt  \
   --weights=/net/liuwenran/compress/pqCompress/output/lenet5/all_stage2_iter_10000.caffemodel 
