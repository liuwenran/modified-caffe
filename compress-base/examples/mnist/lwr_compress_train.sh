#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/lwr_compress_solver.prototxt  --weights=/net/liuwenran/compress/surgery/output/allChanged/lenet_compress_all_reference_iter_10000.caffemodel
