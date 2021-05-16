#!/bin/bash

# input_graph=/home/hnaik/git/ignn/output/sr-alt-20210515-080147.json
# input_graph=/home/hnaik/git/ignn/output/sr-alt-20210515-085240.json
# input_graph=/home/hnaik/git/ignn/output/sr-alt-20210515-101613.json
# input_graph="${input_dir}/sr-alt-20210515-110339.json"

# graph=sr-alt-20210515-112838.json
# graph=sn-count-20210515-211556.json ## x10
graph=sn-count-20210515-221325.json

input_dir="/home/hnaik/git/ignn/output"
input_graph="${input_dir}/${graph}"

explain() {
    node=$1
    threshold_num=$2

    set -x
    python -m explainer_main \
	   --gpu \
	   --dataset=spiked-rings \
	   --explain-node=${node} \
	   --threshold-num=${threshold_num}
    set +x
}

train() {
    rm -rf log/*
    python train.py --dataset=spiked-rings --input-graph=${input_graph}
}

pushd .
set -x
cd ~/git/gnn-model-explainer

explain 3 3

# 1
# explain 77 3

# 2
# explain 97 4
# explain 99 4

# 3
# explain 133 4
# explain 125 4

# 4
# explain 39 5
# explain 63 6

set +x
popd
