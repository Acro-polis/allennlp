#!/usr/bin/env bash


allennlp find-lr "~/allennlp/training_config/csqa_baseline.jsonnet" \
                -f -s "/home/ubuntu/results/baseline/learning_rate" \
                --stopping-factor=4 \
