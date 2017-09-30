#!/usr/bin/env sh

./build/tools/caffe train  \
   --solver=examples/mnist/pqcompress_fc_stage1_solver.prototxt  \
   --weights=/net/liuwenran/compress/pqCompress/output/lenet5/conv_qstage3_iter_501.caffemodel
