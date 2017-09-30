#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt --weights=/net/liuwenran/compress/surgery/output/alexnet/allChanged/alexnet_compress_reference_borrowed.caffemodel --gpu 2
