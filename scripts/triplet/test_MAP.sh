#!/usr/bin/env sh


GLOG_logtostderr=1 ./build/examples/test_net_triplet_MAP.bin prototxt/triplet/triplet_test_query.prototxt prototxt/triplet/triplet_test_database.prototxt ./snapshots/_iter_$1  result/triplet GPU 32 $1 2>&1 | tee test.log 
