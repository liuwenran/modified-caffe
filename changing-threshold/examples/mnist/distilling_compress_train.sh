#!/usr/bin/env sh

./build/tools/caffe train  \
    --solver=examples/mnist/distilling_compress_solver.prototxt  \
    --weights=/home/liuwr/liuwenran/compress/distilling/output/lenetTOT/distilling_compress_0.5_0.5_0.9_0.5_0.5_0.5_0.2_0.2_0.5_0.5_iter_20000.caffemodel
#    --weights=/home/liuwr/liuwenran/compress/distilling/output/lenetTO/lenet5_iter_10000.caffemodel
#    --weights=/home/liuwr/liuwenran/compress/distilling/output/lenetTOT/distilling_compress_0.5_0.5_0.9_0.5_iter_20000.caffemodel
#    --weights=/home/liuwr/liuwenran/compress/distilling/output/lenetTOT/lenetTOT_compress_0.5_0.5_0.9_iter_20000.caffemodel,/home/liuwr/liuwenran/compress/distilling/output/lenetTO/lenet5_iter_10000.caffemodel
