#!/usr/bin/env sh


GLOG_logtostderr=1 ./build/examples/train_net_triplet.bin prototxt/triplet/triplet_solver.prototxt snapshorts/_iter_FIN $1 2>&1 | tee train.log



