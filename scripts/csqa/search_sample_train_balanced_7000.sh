#!/usr/bin/env bash


python search_logical_forms.py --csqa_directory="/home/ubuntu/Desktop/CSQA_v9/train" \
                               --initial_dict="/home/ubuntu/Desktop/CSQA_v9/sample_train_7000_balanced_logical_forms.p" \
                               --wikidata_directory="/home/ubuntu/Desktop/wikidata/wikidata_short_1_2_rev.p" \
                               --wikidata_type_directory="/home/ubuntu/Desktop/wikidata/par_child_dict_full.p" \
                               --instance_limit=7000 \
                               --result_name="_balanced" \
                               --target_sampling=1000

