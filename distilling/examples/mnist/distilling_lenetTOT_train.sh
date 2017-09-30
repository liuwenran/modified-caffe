 #!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/distilling_lenetTOT_solver.prototxt --weights=/home/liuwr/liuwenran/compress/distilling/output/lenetTO/lenet5_iter_10000.caffemodel  

#--weights=/home/liuwr/liuwenran/compress/distilling/output/lenetTO/lenetTO_iter_200000.caffemodel
