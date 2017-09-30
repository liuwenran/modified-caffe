#!/usr/bin/env sh

./build/tools/caffe train  \
   --solver=examples/mnist/pqcompress_all_stage1_solver.prototxt  \
   --weights=/net/liuwenran/compress/pqCompress/output/lenet5/all_stage0_iter_10000.caffemodel 
