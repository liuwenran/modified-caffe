#!/usr/bin/env sh

./build/tools/caffe train  \
   --solver=examples/mnist/pqcompress_all_stage2_solver.prototxt  \
   --weights=/net/liuwenran/compress/pqCompress/output/lenet5/all_stage1_iter_10000.caffemodel 
