#!/usr/bin/env sh

./build/tools/caffe train --solver=models/bvlc_alexnet/solver_noWD.prototxt --gpu 2
