#!/usr/bin/env bash


#for lr in 0.02 0.03 0.04 0.05 0.06
for lr in 2 3 4 5 6
do
    echo "running with lr=$lr"
    export learning_rate=$lr
    allennlp train "~/allennlp/training_config/csqa_baseline_lr_schedule.jsonnet" \
             -f -s "home/ubuntu/results/csqa_baseline/csqa_baseline_triangular_lr="$lr"_cutfrac=0.25"
done
